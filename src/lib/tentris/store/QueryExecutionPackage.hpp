#ifndef TENTRIS_QUERYEXECUTIONPACKAGE_HPP
#define TENTRIS_QUERYEXECUTIONPACKAGE_HPP

#include <any>
#include <exception>
#include <ostream>
#include <sstream>  // for std::ostringstream

#include <sqlite3.h>
#include <vector>
#include <string>
#include <memory>

#include "tentris/store/RDF/TermStore.hpp"
#include "tentris/store/AtomicTripleStore.hpp"
#include "tentris/store/SPARQL/ParsedSPARQL.hpp"
#include "tentris/tensor/BoolHypertrie.hpp"


namespace tentris::store {
	class TripleStore;
};

namespace tentris::store::cache {

	/**
	 * A QueryExecutionPackage contains everything that is necessary to execute a given sparql query for a state of the
	 * RDF graph.
	 */
	struct QueryExecutionPackage {
		using const_BoolHypertrie = ::tentris::tensor::const_BoolHypertrie;
		using time_point_t = logging::time_point_t;
		using SelectModifier = sparql::SelectModifier;
		using Variable = Dice::sparql::Variable;
		using ParsedSPARQL = sparql::ParsedSPARQL;
		using Subscript = ::tentris::tensor::Subscript;
		using Label = Subscript::Label;
		std::shared_ptr<tensor::CardinalityEstimation> cardinalityestimation;
//		std::string filepath = "/home/sohail/CLionProjects/tentris/query-features.csv";
		std::string filepath = "../query-features.csv";
		std::vector<std::string> row;
		std::vector<int> tpSizes;

		sqlite3* db;
		char* errorMessage = 0;
		// Get the auto-incremented ID
		int queryID;


	private:
		std::string sparql_string;
		std::shared_ptr<Subscript> subscript;
		SelectModifier select_modifier;
		std::vector<Variable> query_variables;
		std::shared_ptr<ParsedSPARQL> parsed_sparql;

	public:
		/**
		 * Indicates if the QueryExecutionPackage represents an distinct query or not. If it is distinct use only
		 * the methods with distinct in their names. Otherwise use only the methods with regular in their names
		 */

		bool is_trivial_empty = false;

	private:

		std::vector<const_BoolHypertrie> operands{};

	public:
		QueryExecutionPackage() = delete;
		std::shared_ptr<einsum::internal::Context> contextInstance; // Member to store the Context instance

		/**
		 *
		 * @param sparql_string sparql query to be parsed
		 * @param trie current try holding the data
		 * @param termIndex term store attached to the trie
		 * @throw std::invalid_argument the sparql query was not parsable
		 */
		explicit QueryExecutionPackage(const std::string &sparql_string) : sparql_string{sparql_string} {
			using namespace logging;
			logDebug(fmt::format("Parsing query: {}", sparql_string));

			// Open the SQLite database (or create if it doesn't exist)
			int rc = sqlite3_open("/home/sohail/CLionProjects/tentris/query_data.db", &db);
			if(rc) {
				std::cerr << "Can't open database: " << sqlite3_errmsg(db) << std::endl;
			} else {
				std::cout << "Opened database successfully!" << std::endl;
			}
			// Create a table if it doesn't exist
			std::string createTableSQL = "CREATE TABLE IF NOT EXISTS TestQueryData ("
										 "QueryString TEXT, "
										 "QueryVars TEXT, "
										 "ProjVars TEXT, "
										 "JoinVars TEXT, "
										 "NonJoinVars TEXT, "
										 "MinCardinalityInTP TEXT, "
										 "SelectModifier INT, "
										 "NoTPs INT, "
										 "TPSizes TEXT, "
										 "VartoLabelMap TEXT, "  // New column for variable-to-label map
										 "QueryPlan TEXT, "
										 "TentrisQueryRuntime REAL, "
										 "DRLQueryRuntime REAL);";

			rc = sqlite3_exec(db, createTableSQL.c_str(), nullptr, nullptr, &errorMessage);
			if(rc != SQLITE_OK) {
				std::cerr << "SQL error: " << errorMessage << std::endl;
				sqlite3_free(errorMessage);
			} else {
				std::cout << "Table created successfully!" << std::endl;
			}
			//ParsedSPARQL parsed_sparql{sparql_string};
			parsed_sparql = std::make_shared<ParsedSPARQL>(sparql_string);
			subscript = parsed_sparql->getSubscript();
			//select_modifier = parsed_sparql->getSelectModifier();
			logDebug(fmt::format("Parsed subscript: {} [distinct = {}]",
								 subscript,
								 select_modifier == SelectModifier::DISTINCT));
			query_variables = parsed_sparql->getQueryVariables();

			auto &triple_store = AtomicTripleStore::getInstance();

			logDebug(fmt::format("Slicing TPs"));
			for ([[maybe_unused]] const auto &[op_pos, tp]: iter::enumerate(parsed_sparql->getBgps())) {
				logDebug(fmt::format("Slice key {}: ⟨{}⟩", op_pos, fmt::join(tp, ", ")));
				std::variant<const_BoolHypertrie, bool> op = triple_store.resolveTriplePattern(tp);
				if (std::holds_alternative<bool>(op)) {
					is_trivial_empty = not std::get<bool>(op);
					tpSizes.push_back(0); // Save 0 for this op
					logTrace(fmt::format("Operand {} is {}", op_pos, is_trivial_empty));
				} else {
					auto bht = std::get<const_BoolHypertrie>(op);
					if (not bht.empty()) {
						tpSizes.push_back(bht.size()); // Save the size returned
						logTrace(fmt::format("Operand {} size {}", op_pos, bht.size()));
						operands.emplace_back(bht);
					} else {
						is_trivial_empty = true;
						operands.clear();
					}
				}
				if (is_trivial_empty) {
					logDebug(fmt::format("Query is trivially empty, i.e. the lastly sliced operand {} is emtpy.", op_pos));
					break;
				}
			}
			writeToSQLite(parsed_sparql, db, tpSizes);
			contextInstance = std::make_shared<einsum::internal::Context>(queryID);
		}

	private:
		/**
		 * Builds the operator tree for this query.
		 * @tparam RESULT_TYPE the type returned by the operand tree
		 * @param slice_keys slice keys to extract the operands from the hypertries. slice_keys and hypertries must be
		 * of equal length.
		 * @param subscript the subscript that spans the operator tree.
		 * @param hypertries a list of hypertries. typically this is a list containing the data base hypertrie multiple
		 * times.
		 * @return
		 */
		template<typename RESULT_TYPE>
		static std::shared_ptr<void> generateEinsum(const std::shared_ptr<Subscript> &subscript,
													const std::vector<const_BoolHypertrie> &hypertries,
													const time_point_t &timeout,
													const std::shared_ptr<einsum::internal::Context>& contextInstance) {
			using namespace tensor;
			return std::make_shared<Einsum<RESULT_TYPE>>(subscript, hypertries, timeout, contextInstance);
		}

	public:
		std::shared_ptr<void> getEinsum(const time_point_t &timeout = time_point_t::max()) const {
			using namespace tensor;
			//std::cout << "the timeout value in getEinsum func in QueryExecutionPackage is " << timeout << std::endl;
			if (select_modifier == SelectModifier::NONE)
				return generateEinsum<COUNTED_t>(subscript, operands, timeout, contextInstance);
			else
				return generateEinsum<DISTINCT_t>(subscript, operands, timeout, contextInstance);
		}

		const std::string &getSparqlStr() const {
			return sparql_string;
		}


		const std::vector<Variable> &getQueryVariables() const {
			return query_variables;
		}

		friend struct ::fmt::formatter<QueryExecutionPackage>;

		//code by sohail started here

		// Getters for the features to expose to Python DRL gym env.
		std::vector<std::string> getQueryVars() const {
			std::vector<std::string> vars;
			for (const auto& var : parsed_sparql->getVariables()) {
				vars.push_back(var.getName());
			}
			return vars;
		}
		std::vector<std::string> getProjVars() const {
			std::vector<std::string> projVars;
			for (const auto& var : parsed_sparql->getQueryVariables()) {
				projVars.push_back(var.getName());
			}
			return projVars;
		}
		std::vector<std::string> getJoinVars() const {
			std::vector<std::string> joinVars;
			for (const auto& var : parsed_sparql->getJoinVariables()) {
				joinVars.push_back(var.getName());
			}
			return joinVars;
		}
		std::vector<std::string> getNonJoinVars() const {
			std::vector<std::string> nonJoinVars;
			for (const auto& var : parsed_sparql->get_non_join_vars()) {
				nonJoinVars.push_back(var.getName());
			}
			return nonJoinVars;
		}
		std::vector<double> getMinCardinalityInTP() const {
			std::vector<double> minimum_cardinalities_labels;
			for (const auto& [var, lab] : parsed_sparql->var_to_label) {
				std::vector<double> all_card_of_a_label = cardinalityestimation->calcCardPublicInterface(operands, lab, subscript);
				if (!all_card_of_a_label.empty()) {
					auto min_cards = std::min_element(all_card_of_a_label.begin(), all_card_of_a_label.end());
					minimum_cardinalities_labels.push_back(*min_cards);
				} else {
					// If all_card_of_a_label is unexpectedly empty, save a value of 0
					minimum_cardinalities_labels.push_back(0.0);
				}
			}
			return minimum_cardinalities_labels;
		}
		int getSelectModifier() const {
			return static_cast<int>(select_modifier);
		}
		int getNoTPs() const {
			return parsed_sparql->getBgps().size();
		}
		std::vector<int> getTPSizes() const {
			return tpSizes;
		}
		std::string getVartoLabelMap() {
			return this->convertMapToString(parsed_sparql->var_to_label);  // Placeholder
		}

		std::string convertMapToString(const robin_hood::unordered_map<Variable, Label>& var_to_label) {
			std::ostringstream oss;
			oss << "{";

			bool first = true;
			for (const auto& pair : var_to_label) {
				if (!first) {
					oss << ", ";  // Add comma and space between elements
				}
				oss << "\"" << pair.first.getName() << "\": \"" << pair.second << "\"";  // Format as "key": "value"
				first = false;
			}

			oss << "}";
			return oss.str();
		}

		void writeToSQLite(const std::shared_ptr<ParsedSPARQL>& parsed_sparql, sqlite3* db, std::vector<int> tpSizes) {
			// Step 1: Check if the record already exists based on QueryString
			int rc;
			std::string checkSQL = "SELECT rowid FROM TestQueryData WHERE QueryString = ?;";

			sqlite3_stmt* checkStmt;
			rc = sqlite3_prepare_v2(db, checkSQL.c_str(), -1, &checkStmt, nullptr);

			if (rc != SQLITE_OK) {
				std::cerr << "Failed to prepare check statement: " << sqlite3_errmsg(db) << std::endl;
				return;
			}

			// Bind the QueryString to check
			sqlite3_bind_text(checkStmt, 1, getSparqlStr().c_str(), -1, SQLITE_TRANSIENT);

			// Step 2: Execute the query to see if the record exists
			rc = sqlite3_step(checkStmt);
			if (rc == SQLITE_ROW) {
				// Record exists, retrieve queryID
				queryID = sqlite3_column_int(checkStmt, 0); // Fetch the queryID (rowid) of the existing record
				std::cout << "Record already exists with queryID: " << queryID << std::endl;

				// Finalize the statement and return since the record already exists
				sqlite3_finalize(checkStmt);
				return;
			}

			// Finalize the check statement since record doesn't exist
			sqlite3_finalize(checkStmt);

			//
			// Step 3: Insert the record if it doesn't exist
			std::string insertSQL = "INSERT INTO TestQueryData (QueryString, QueryVars, ProjVars, JoinVars, NonJoinVars, "
									"MinCardinalityInTP, SelectModifier, NoTPs, TPSizes, VartoLabelMap, QueryPlan, "
									"TentrisQueryRuntime, DRLQueryRuntime) "
									"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";

			sqlite3_stmt* stmt;
			rc = sqlite3_prepare_v2(db, insertSQL.c_str(), -1, &stmt, nullptr);

			if (rc != SQLITE_OK) {
				std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(db) << std::endl;
				return;
			}

			// Bind data to the statement (replace with your logic for extracting and binding query features)
			sqlite3_bind_text(stmt, 1, getSparqlStr().c_str(), -1, SQLITE_TRANSIENT);  // QueryString

			// Bind QueryVars
			std::string queryVarsStr;
			for (const auto& var : parsed_sparql->getVariables()) {
				queryVarsStr += var.getName() + " ";
			}
			sqlite3_bind_text(stmt, 2, queryVarsStr.c_str(), -1, SQLITE_TRANSIENT);


			// Bind ProjVars, JoinVars, Non-JoinVars, etc. similar to how you did for CSV
			// Example:
			auto projVars = parsed_sparql->getQueryVariables(); //adding projection variables to the database
			std::string projVarsStr;
			for (const auto& var : projVars) {
				projVarsStr += var.getName() + " ";
			}
			if (!projVarsStr.empty()) {
				projVarsStr.pop_back(); // Remove trailing space
			}
				sqlite3_bind_text(stmt, 3, projVarsStr.c_str(), -1, SQLITE_TRANSIENT);

			std::string joinVarStr;	//adding join variables to the database
			auto joinVars = parsed_sparql->getJoinVariables();
			for (const auto& var : joinVars) {
				joinVarStr += var.getName() + " ";
			}
			if (!joinVarStr.empty()) {
				joinVarStr.pop_back(); // Remove trailing space
			}
			sqlite3_bind_text(stmt, 4, joinVarStr.c_str(), -1, SQLITE_TRANSIENT);

			std::string non_join_VarStr;	//adding non_join variables to the database
			auto non_join_Vars = parsed_sparql->get_non_join_vars();
			for (const auto& var : non_join_Vars) {
				non_join_VarStr += var.getName() + " ";
			}
			if (!non_join_VarStr.empty()) {
				non_join_VarStr.pop_back(); // Remove trailing space
			}
			sqlite3_bind_text(stmt, 5, non_join_VarStr.c_str(), -1, SQLITE_TRANSIENT);

			// Add cardinalities for each label
			std::string min_var_CardsInTP;
			for (const auto& [var, lab] : parsed_sparql->var_to_label) {
				std::vector<double> cardinalities = cardinalityestimation->calcCardPublicInterface(operands, lab, subscript);
				auto min_cardinality = std::min_element(cardinalities.begin(), cardinalities.end());
				min_var_CardsInTP += std::to_string(*min_cardinality) + " ";
			}
			if (!min_var_CardsInTP.empty()) {
				min_var_CardsInTP.pop_back(); // Remove trailing space and semicolon
			}
			sqlite3_bind_text(stmt, 6, min_var_CardsInTP.c_str(), -1, SQLITE_TRANSIENT);

			// Add the select_modifier
			std::string select_mod;
			if (select_modifier == SelectModifier::NONE) {
				select_mod = "0";  // Push "0" as a string
			} else if (select_modifier == SelectModifier::DISTINCT) {
				select_mod = "1"; // Push "1" as a string
			} else {
				select_mod = "2";  // Push "2" as a string
			}
			sqlite3_bind_text(stmt, 7, select_mod.c_str(), -1, SQLITE_TRANSIENT);

			// Add the number of TPs
			sqlite3_bind_int(stmt, 8, (parsed_sparql->getBgps().size()));

			// Add TP-sizes to the row
			std::string tpSizesStr;
			for (const int size : tpSizes) {
				tpSizesStr += std::to_string(size) + " ";
			}
			if (!tpSizesStr.empty()) {
				tpSizesStr.pop_back(); // Remove trailing space
			}
			sqlite3_bind_text(stmt, 9, tpSizesStr.c_str(), -1, SQLITE_TRANSIENT);

			// Bind VartoLabelMap using convertMapToString
			std::string varToLabelStr = convertMapToString(parsed_sparql->var_to_label);
			sqlite3_bind_text(stmt, 10, varToLabelStr.c_str(), -1, SQLITE_TRANSIENT);

			// Execute the statement
			rc = sqlite3_step(stmt);
			if (rc != SQLITE_DONE) {
				std::cerr << "Failed to insert data: " << sqlite3_errmsg(db) << std::endl;
			} else {
				queryID = sqlite3_last_insert_rowid(db); // Retrieve the queryID of the inserted row
				std::cout << "Record inserted with queryID: " << queryID << std::endl;
			}
			// Cleanup
			sqlite3_finalize(stmt);
		}

		void writeToCSV(const std::shared_ptr<ParsedSPARQL>& parsed_sparql, const std::string& filepath, std::vector<int> tpSizes) {
			CSVWriter csvWriter(filepath); // Create a CSV writer that will write to the given file.

			// Write the header
//			csvWriter.writeRow({"query", "var-to-lab-mapping", "lab-cards-in-TP", "ProjVars", "QueryVars", "LonelyNonResultLabels", "No.TPs", "TP-sizes", "QueryRuntime" });
			csvWriter.writeRow({ "QueryString", "QueryVars", "ProjVars", "JoinVars", "LonelyNonResultLabels", "min_cardinality_of_var_in_TP", "select_modifier",  "No.TPs", "TP-sizes", "QueryRuntime" });

//			std::vector<std::string> row;
			row.push_back(getSparqlStr());
			// Iterate over the map and write each key-value pair to the CSV

			// Add QueryVars to the row
			auto queryVars = parsed_sparql->getVariables();
			std::string queryVarsStr;
			for (const auto& var : queryVars) {
				queryVarsStr += var.getName() + " ";
			}
			if (!queryVarsStr.empty()) {
				queryVarsStr.pop_back(); // Remove trailing space
			}
			row.push_back(queryVarsStr);


//			// Add the var-to-lab-mapping to the row
			std::string varToLabMapping;
			for (const auto& [var, lab] : parsed_sparql->var_to_label) {
				varToLabMapping += var.getName() + ":" + std::string(1, lab) + " ";
			}
			if (!varToLabMapping.empty()) {
				varToLabMapping.pop_back(); // Remove trailing space
			}
//			row.push_back(varToLabMapping);

			auto projVars = parsed_sparql->getQueryVariables();
			// Add labels for ProjVars to the row
			std::string projVarsLabelsStr;
			for (const auto& var : projVars) {
				projVarsLabelsStr += var.getName() + " ";
//				if (parsed_sparql->var_to_label.find(var) != parsed_sparql->var_to_label.end()) {
//					projVarsLabelsStr += std::string(1, parsed_sparql->var_to_label.at(var)) + " ";}
			}
			if (!projVarsLabelsStr.empty()) {
				projVarsLabelsStr.pop_back(); // Remove trailing space
			}
			row.push_back(projVarsLabelsStr);

			// Add Join-Variables
			std::string joinVarStr;
			auto joinVars = parsed_sparql->getJoinVariables();
			for (const auto& var : joinVars) {
				std::cout << "The join variables are " << var.getName() << "\t";
//				if (parsed_sparql->var_to_label.find(var) != parsed_sparql->var_to_label.end()) {
//					joinVarStr += std::string(1, parsed_sparql->var_to_label.at(var)) + " ";}
				joinVarStr += var.getName() + " ";
			}
			if (!joinVarStr.empty()) {
				joinVarStr.pop_back(); // Remove trailing space
			}
			row.push_back(joinVarStr);


			std::unordered_set<Variable> lonelyNonResultVarsSet;

			for (const auto& var : queryVars) {
				// Check if var is not in projVars and not in joinVars
				if (std::find(projVars.begin(), projVars.end(), var) == projVars.end() &&
					std::find(joinVars.begin(), joinVars.end(), var) == joinVars.end()) {
					lonelyNonResultVarsSet.insert(var); // Add var to lonelyNonResultVarsSet
				}
			}

			// Now convert the lonelyNonResultVarsSet to a string to add to the row
			std::string lonelyNonResultVarsStr;
			for (const auto& var : lonelyNonResultVarsSet) {
				lonelyNonResultVarsStr += var.getName() + " ";
			}

			if (!lonelyNonResultVarsStr.empty()) {
				lonelyNonResultVarsStr.pop_back(); // Remove trailing space
			}

			row.push_back(lonelyNonResultVarsStr);  // Add LonelyNonResultVars to the row



			// Add cardinalities for each label
			std::string varCardsInTP;
			for (const auto& [var, lab] : parsed_sparql->var_to_label) {
				std::vector<double> cardinalities = cardinalityestimation->calcCardPublicInterface(operands, lab, subscript);
				auto min_cardinality = std::min_element(cardinalities.begin(), cardinalities.end());
//				varCardsInTP += std::string(1, lab) + ":";
//				std::ostringstream oss;
//				oss.precision(2);
//				oss << std::fixed;
//				for (double card : cardinalities) {
//					oss << card << " ";
//				}
//				std::string cardStr = oss.str();
//				if (!cardStr.empty()) {
//					cardStr.pop_back(); // Remove trailing space
//				}
				varCardsInTP += std::to_string(*min_cardinality) + " ";
			}
			if (!varCardsInTP.empty()) {
				varCardsInTP.pop_back(); // Remove trailing space and semicolon
			}
			row.push_back(varCardsInTP);

			// Add the select_modifier
			if (select_modifier == SelectModifier::NONE) {
				row.push_back("0");  // Push "0" as a string
			} else if (select_modifier == SelectModifier::DISTINCT) {
				row.push_back("1");  // Push "1" as a string
			} else {
				row.push_back("2");  // Push "2" as a string
			}


			// Add the count of bgps instead of their serialized forms
			row.push_back(std::to_string(parsed_sparql->getBgps().size()));

			// Add TP-sizes to the row
			std::string tpSizesStr;
			for (const int size : tpSizes) {
				tpSizesStr += std::to_string(size) + " ";
			}
			if (!tpSizesStr.empty()) {
				tpSizesStr.pop_back(); // Remove trailing space
			}
			row.push_back(tpSizesStr);
			}
		//code by sohail ended here

	};
} // namespace tentris::store::cache

template<>
struct fmt::formatter<tentris::store::cache::QueryExecutionPackage> {
	template<typename ParseContext>
	constexpr auto parse(ParseContext &ctx) { return ctx.begin(); }

	template<typename FormatContext>
	auto format(const tentris::store::cache::QueryExecutionPackage &p, FormatContext &ctx) {
		using SelectModifier = tentris::store::sparql::SelectModifier;
		return format_to(ctx.begin(),
						 " SPARQL:     {}\n"
						 " subscript:  {}\n"
						 " is_distinct:      {}\n"
						 " is_trivial_empty: {}\n",
						 p.sparql_string, p.subscript, p.select_modifier == SelectModifier::DISTINCT,
						 p.is_trivial_empty);
	}
};

#endif // TENTRIS_QUERYEXECUTIONPACKAGE_HPP

