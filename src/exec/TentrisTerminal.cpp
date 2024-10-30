#include <string>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <thread>

#include <tentris/store/QueryExecutionPackage.hpp>
#include <tentris/store/QueryExecutionPackageCache.hpp>
#include <tentris/store/TripleStore.hpp>
#include <tentris/util/LogHelper.hpp>
#include <tentris/tensor/BoolHypertrie.hpp>
#include <tentris/http/QueryResultState.hpp>

#include <fmt/core.h>
#include <fmt/format.h>
#include <itertools.hpp>
#include <sqlite3.h>


#include "config/TerminalConfig.hpp"
#include "VersionStrings.hpp"

using namespace tentris::store;
using namespace tentris::logging;
using namespace tentris::store::cache;
using namespace tentris::store::sparql;
using namespace std::filesystem;
using namespace iter;
using namespace tentris::tensor;
using namespace std::chrono;

using Variable = Dice::sparql::Variable;

TerminalConfig cfg;

bool onlystdout = false;

using Errors = tentris::http::ResultState;

std::ostream &logsink() {
	if (onlystdout)
		return std::cout;
	else
		return std::cerr;
}

inline std::string tp2s(time_point_t timepoint) {
	auto in_time_t = system_clock::to_time_t(
			system_clock::now() + duration_cast<system_clock::duration>(timepoint - steady_clock::now()));

	std::stringstream ss;
	ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
	return ss.str();
};

time_point_t query_start;
time_point_t query_end;
time_point_t parse_start;
time_point_t parse_end;
time_point_t execute_start;
time_point_t execute_end;

Errors error;
size_t number_of_bindings;

time_point_t timeout;
time_point_t actual_timeout;

template<typename RESULT_TYPE>
void
writeNTriple(std::ostream &stream, const std::shared_ptr<QueryExecutionPackage> &query_package) {
	const std::vector<Variable> &vars = query_package->getQueryVariables();
	stream << fmt::format("{}\n", fmt::join(vars, ","));

	uint timeout_check = 0;
	size_t result_count = 0;

	bool first = true;

	if (not query_package->is_trivial_empty) {
		std::shared_ptr<void> raw_results = query_package->getEinsum(timeout);
		auto &results = *static_cast<Einsum<RESULT_TYPE> *>(raw_results.get());
		for (const auto &result : results) {
			if (first) {
				first = false;
				execute_end = steady_clock::now();
			}

			std::stringstream ss;
			bool inner_first = true;
			for (auto binding : result.key) {
				if (inner_first)
					inner_first = false;
				else
					ss << ",";
				if (binding != nullptr)
					ss << binding->getIdentifier();
			}
			ss << "\n";

			std::string binding_string = ss.str();

			for ([[maybe_unused]] const auto c : iter::range(result.value)) {
				stream << binding_string;
				++result_count;
				if (++timeout_check == 500) {
					timeout_check = 0;
					stream.flush();
					if (auto current_time = steady_clock::now(); current_time > timeout) {
						::error = Errors::SERIALIZATION_TIMEOUT;
						actual_timeout = current_time;
						number_of_bindings = result_count;
						return;
					}
				}
			}
		}
	}
	if (first) { // if no bindings are returned
		execute_end = steady_clock::now();
	}
	number_of_bindings = result_count;
}

template<typename RESULT_TYPE>
inline void runCMDQuery(const std::shared_ptr<QueryExecutionPackage> &query_package,
						const time_point_t timeout) {
	// calculate the result
	// check if it timed out
	if (steady_clock::now() < timeout) {
		writeNTriple<RESULT_TYPE>(std::cout, query_package);
	} else {
		::error = Errors::PROCESSING_TIMEOUT;
		actual_timeout = steady_clock::now();
	}
}

void insertTentrisQueryRuntime(sqlite3* db, int queryID, long long tentrisExecutionTime) {
	char* errorMessage = 0;

	// SQL update query for TentrisQueryRuntime
	std::string updateSQL = "UPDATE TestQueryData SET TentrisQueryRuntime = ? WHERE rowid = ?;";

	sqlite3_stmt* stmt;
	int rc = sqlite3_prepare_v2(db, updateSQL.c_str(), -1, &stmt, nullptr);
	if (rc != SQLITE_OK) {
		std::cerr << "Failed to prepare update statement for TentrisQueryRuntime: " << sqlite3_errmsg(db) << std::endl;
		return;
	}

	// Bind TentrisQueryRuntime and QueryID
	sqlite3_bind_int64(stmt, 1, tentrisExecutionTime);
	sqlite3_bind_int(stmt, 2, queryID);

	// Execute the update
	rc = sqlite3_step(stmt);
	if (rc != SQLITE_DONE) {
		std::cerr << "Failed to update TentrisQueryRuntime: " << sqlite3_errmsg(db) << std::endl;
	}

	// Clean up
	sqlite3_finalize(stmt);
}

void insertDRLQueryRuntime(sqlite3* db, int queryID, long long drlExecutionTime) {
	char* errorMessage = 0;

	// SQL update query for DRLQueryRuntime
	std::string updateSQL = "UPDATE TestQueryData SET DRLQueryRuntime = ? WHERE rowid = ?;";

	sqlite3_stmt* stmt;
	int rc = sqlite3_prepare_v2(db, updateSQL.c_str(), -1, &stmt, nullptr);
	if (rc != SQLITE_OK) {
		std::cerr << "Failed to prepare update statement for DRLQueryRuntime: " << sqlite3_errmsg(db) << std::endl;
		return;
	}

	// Bind DRLQueryRuntime and QueryID
	sqlite3_bind_int64(stmt, 1, drlExecutionTime);
	sqlite3_bind_int(stmt, 2, queryID);

	// Execute the update
	rc = sqlite3_step(stmt);
	if (rc != SQLITE_DONE) {
		std::cerr << "Failed to update DRLQueryRuntime: " << sqlite3_errmsg(db) << std::endl;
	}

	// Clean up
	sqlite3_finalize(stmt);
}

long long fetchQueryRuntimeFromSQLite(sqlite3* db, int queryID) {
	std::string selectSQL = "SELECT TentrisQueryRuntime FROM TestQueryData WHERE rowid = ?;";
	sqlite3_stmt* stmt;
	long long queryRuntime = 0;

	int rc = sqlite3_prepare_v2(db, selectSQL.c_str(), -1, &stmt, nullptr);
	if (rc != SQLITE_OK) {
		std::cerr << "Failed to prepare select statement: " << sqlite3_errmsg(db) << std::endl;
		return queryRuntime;
	}

	// Bind the query ID to the statement
	sqlite3_bind_int(stmt, 1, queryID);

	// Execute the statement and fetch the QueryRuntime
	if (sqlite3_step(stmt) == SQLITE_ROW) {
		queryRuntime = sqlite3_column_int64(stmt, 0); // Get the QueryRuntime
	} else {
		std::cerr << "Failed to fetch QueryRuntime: " << sqlite3_errmsg(db) << std::endl;
	}

	// Clean up
	sqlite3_finalize(stmt);
	std::cout << "Tentris Query Runtime fetched from the db is " << queryRuntime << std::endl;
	return queryRuntime;
}


// Function to read SPARQL queries from the SQLite database
std::vector<std::pair<int, std::string>> readSparqlQueriesFromDatabase(sqlite3* db) {
	std::vector<std::pair<int, std::string>> queries;
	std::string selectSQL = "SELECT rowid, QueryString FROM TestQueryData;"; // Fetch rowid and QueryString from the table

	sqlite3_stmt* stmt;
	int rc = sqlite3_prepare_v2(db, selectSQL.c_str(), -1, &stmt, nullptr);
	if (rc != SQLITE_OK) {
		std::cerr << "Failed to prepare select statement: " << sqlite3_errmsg(db) << std::endl;
		return queries;
	}

	// Execute the statement and fetch each row
	while (sqlite3_step(stmt) == SQLITE_ROW) {
		int queryID = sqlite3_column_int(stmt, 0); // Get the rowid
		const char* queryText = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)); // Get the SPARQL query string
		if (queryText) {
			queries.emplace_back(queryID, std::string(queryText));
		}
	}

	// Clean up
	sqlite3_finalize(stmt);
	return queries;
}

std::vector<std::string> readSparqlQueriesFromCsv(const std::string &csv_filepath) {
	std::vector<std::string> queries;
	std::ifstream file(csv_filepath);
	if (!file.is_open()) {
		logsink() << fmt::format("Error: Could not open file {}\n", csv_filepath);
		return queries;
	}

	// Skip the first line (title of the column)
	std::string line;
	if (!std::getline(file, line)) {
		logsink() << fmt::format("Error: Failed to read the first line (title) from {}\n", csv_filepath);
		return queries;
	}

	// Read each subsequent line (SPARQL queries)
	while (std::getline(file, line)) {
		if (!line.empty()) {
			logsink() << fmt::format("Read query: {}\n", line);  // Log the query
			queries.push_back(line);
		}
	}

	file.close();
	return queries;
}


void commandlineInterface(QueryExecutionPackage_cache& querypackage_cache, sqlite3* db, std::string file_path) {
	std::shared_ptr<QueryExecutionPackage> query_package;


	// Read SPARQL queries from the database
//	std::vector<std::pair<int, std::string>> sparql_queries = readSparqlQueriesFromDatabase(db);

	//Read SPARQL queries from csv file
	std::vector<std::string> sparql_queries = readSparqlQueriesFromCsv(file_path);
//	for (const auto& [queryID, sparql_str] : sparql_queries) {	//uncomment when loading queries from database
	for (const auto& sparql_str : sparql_queries) {	//uncomment when loading queries from csv file
		try {
//			logsink() << fmt::format("Processing query (ID: {}): {}\n", queryID, sparql_str);	//uncomment when loading queries from database
			logsink() << fmt::format("Processing query: {}\n",  sparql_str); //uncomment when loading queries from csv file

			query_start = steady_clock::now();
			number_of_bindings = 0;
			::error = Errors::OK;

			try {
				parse_start = steady_clock::now();
				query_package = querypackage_cache[sparql_str];
				// Fetch QueryRuntime from the database
//				long long queryRuntime = fetchQueryRuntimeFromSQLite(db, queryID); //uncomment when loading queries from database
				//uncomment the below if else part if loading queries from database
//				// If no runtime is available, use the default timeout (if required)
//				if (queryRuntime == 0) {
//					timeout = steady_clock::now() + cfg.timeout;
//				}
//				else {
//					// Set timeout to 2 times the QueryRuntime
//					timeout = steady_clock::now() + std::chrono::nanoseconds(static_cast<long long>(queryRuntime * 3));
//					// Calculate the time duration between now and the timeout value
//					auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(timeout - steady_clock::now()).count();
//					std::cout << "The timeout for this query is " << duration_ns << " nanoseconds from now." << std::endl;
//
//				}
				timeout = steady_clock::now() + cfg.timeout; //tentris default

				parse_end = steady_clock::now();
				execute_start = steady_clock::now();

				switch (query_package->getSelectModifier()) {
					case SelectModifier::NONE: {
						runCMDQuery<COUNTED_t>(query_package, timeout);
						break;
					}
					case SelectModifier::REDUCE:
						[[fallthrough]];
					case SelectModifier::DISTINCT: {
						runCMDQuery<DISTINCT_t>(query_package, timeout);
						break;
					}
					default:
						break;
				}
			} catch (const std::invalid_argument& e) {
				::error = Errors::UNPARSABLE;
				logDebug(fmt::format("UNPARSABLE reason: {}", e.what()));
			} catch (const std::exception& e) {
				::error = Errors::UNEXPECTED;
				logDebug(fmt::format("UNEXPECTED reason: {}", e.what()));
			} catch (...) {
				::error = Errors::SEVERE_UNEXPECTED;
			}

			query_end = steady_clock::now();

			auto parsing_time = duration_cast<std::chrono::nanoseconds>(parse_end - parse_start);
			auto execution_time = duration_cast<std::chrono::nanoseconds>(execute_end - execute_start);
			auto total_time = duration_cast<std::chrono::nanoseconds>(query_end - query_start);
			auto serialization_time = total_time - execution_time - parsing_time;

			// Insert execution time into SQLite
			insertTentrisQueryRuntime(db, query_package->queryID, execution_time.count());
//			insertDRLQueryRuntime(db, query_package->queryID, execution_time.count());

			switch (::error) {
				case Errors::OK:
					logsink() << "SUCCESSFUL\n";
					break;
				case Errors::UNPARSABLE:
					logsink() << "ERROR: UNPARSABLE QUERY\n";
					break;
				case Errors::PROCESSING_TIMEOUT:
					logsink() << "ERROR: TIMEOUT DURING PROCESSING\n";
					break;
				case Errors::SERIALIZATION_TIMEOUT:
					logsink() << "ERROR: TIMEOUT DURING SERIALIZATION\n";
					break;
				case Errors::UNEXPECTED:
					logsink() << "ERROR: UNEXPECTED\n";
					break;
				case Errors::SEVERE_UNEXPECTED:
					logsink() << "ERROR: SEVERE UNEXPECTED\n";
					break;
				default:
					break;
			}

			logsink() << fmt::format("start: {}\n", tp2s(query_start));
			logsink() << fmt::format("planned timeout: {}\n", tp2s(timeout));
			if (::error == Errors::PROCESSING_TIMEOUT || ::error == Errors::SERIALIZATION_TIMEOUT)
				logsink() << fmt::format("actual timeout: {}\n", tp2s(actual_timeout));
			logsink() << fmt::format("end: {}\n", tp2s(query_end));

			if (::error == Errors::OK || ::error == Errors::PROCESSING_TIMEOUT ||
				::error == Errors::SERIALIZATION_TIMEOUT) {
				logsink() << "number of bindings: " << fmt::format("{:18}", number_of_bindings) << "\n";
				logsink() << "parsing time:       " << fmt::format("{:18}", parsing_time.count()) << " ns\n";
				logsink() << "execution time:     " << fmt::format("{:18}", execution_time.count()) << " ns\n";
				if (::error != Errors::PROCESSING_TIMEOUT)
					logsink() << "serialization time: " << fmt::format("{:18}", serialization_time.count()) << " ns\n";
			}

			logsink() << "total time: " << fmt::format("{:18}", total_time.count()) << " ns\n";
			logsink() << "total time: "
					  << fmt::format("{:12}", duration_cast<std::chrono::milliseconds>(total_time).count())
					  << " ms\n";

			logsink().flush();
		} catch (const std::exception& e) {
			logsink() << "Critical Error: " << e.what() << ". Continuing with the next query.\n";
		} catch (...) {
			logsink() << "Critical Error: Unknown exception. Continuing with the next query.\n";
		}

	}
	std::raise(SIGINT);
}


int main(int argc, char *argv[]) {
	cfg = TerminalConfig{argc, argv};
	tentris::logging::init_logging(cfg.logstdout, cfg.logfile, cfg.logfiledir, cfg.loglevel);

	logsink() << fmt::format("Running {} with {}", tentris_version_string, hypertrie_version_string) << std::endl;

	TripleStore triplestore{};

	QueryExecutionPackage_cache executionpackage_cache{cfg.cache_size};


	onlystdout = cfg.onlystdout;

	if (not cfg.rdf_file.empty()) {
		logsink() << "Loading file " << cfg.rdf_file << " ..." << std::endl;
		auto start_time = steady_clock::now();
		AtomicTripleStore::getInstance().bulkloadRDF(cfg.rdf_file, cfg.bulksize);
		auto duration = steady_clock::now() - start_time;
		logsink() << fmt::format("... loading finished. {} triples loaded.", AtomicTripleStore::getInstance().size())
				  << std::endl;
		logsink() << "duration: {} h {} min {} s"_format(
				(duration_cast<hours>(duration) % 24).count(),
				(duration_cast<minutes>(duration) % 60).count(),
				(duration_cast<seconds>(duration) % 60).count()) << std::endl;
	}

	//path to csv file containing queries
	std::string file_path = "/home/sohail/CLionProjects/tentris/onequery.csv";

	// Path to your SQLite database
	std::string db_path = "/home/sohail/CLionProjects/tentris/query_data.db";
	sqlite3* db;

	// Open the SQLite database
	if (sqlite3_open(db_path.c_str(), &db)) {
		std::cerr << "Can't open database: " << sqlite3_errmsg(db) << std::endl;
		return 1;
	}

	// Use the SQLite database
	std::thread commandline_client{[&executionpackage_cache, db, file_path]() {
		commandlineInterface(executionpackage_cache, db, file_path);
	}};


	// wait for keyboard interrupt
	while (true) {
		sigset_t wset;
		sigemptyset(&wset);
		sigaddset(&wset, SIGINT);
		int number;

		if (int status = sigwait(&wset, &number); status != 0) {
			log("Set contains an invalid signal number.");
			break;
		}
		if (number == SIGINT) {
			logDebug("Exiting by Signal {}."_format(strsignal(number)));
			break;
		}
	}
}
