#include "dice/endpoint/SparqlEndpoint.hpp"

#include <dice/endpoint/TimeoutCheck.hpp>
#include <nlohmann/json.hpp>

#include <dice/query/OperandDependencyGraph.hpp>
#include <spdlog/spdlog.h>
#include <chrono> // For measuring execution time
#include <curl/curl.h>
#include <string>
#include <iostream>
#include <dice/query/operators/CardinalityEstimation.hpp>

#include <dice/endpoint/ParseSPARQLQueryParam.hpp>
#include <dice/endpoint/SparqlJsonResultSAXWriter.hpp>

namespace dice::endpoint {
    std::string QueryFeatures::to_json() const {
        nlohmann::json j;

        // Directly add basic query info
        j["variable_names"] = variable_names;
        j["projection_variables"] = projection_variables;
        j["join_variables"] = join_variables;
        j["non_join_variables"] = non_join_variables;
        j["num_triple_patterns"] = num_triple_patterns;
        j["is_distinct"] = is_distinct;

        // Directly add cardinality features
        j["variable_cardinalities"] = variable_cardinalities;
        j["total_query_cardinality"] = total_query_cardinality;
        j["min_cardinality_variable"] = std::string(1, min_cardinality_variable);

        // Directly add graph features
        j["adjacency_matrix"] = adjacency_matrix;
        j["variable_degrees"] = variable_degrees;
        j["max_variable_degree"] = max_variable_degree;
        j["min_variable_degree"] = min_variable_degree;
        j["avg_variable_degree"] = avg_variable_degree;
        j["graph_density"] = graph_density;
        j["num_connected_components"] = num_connected_components;


        return j.dump();
    }

    SPARQLEndpoint::SPARQLEndpoint(tf::Executor &executor,
                                   triple_store::TripleStore &triplestore,
                                   SparqlQueryCache &sparql_query_cache,
                                   EndpointCfg const &endpoint_cfg)
        : Endpoint(executor, triplestore, sparql_query_cache, endpoint_cfg) {}

    QueryFeatures SPARQLEndpoint::extract_query_features(const sparql2tensor::SPARQLQuery &sparql_query,
                                                         std::chrono::steady_clock::time_point timeout) {
        QueryFeatures features;

        // Basic query information
        for (const auto &[var, var_id] : sparql_query.var_to_id_) {
            features.variable_names.push_back(std::string(var.name()));
        }

        for (const auto &var : sparql_query.projected_variables_) {
            features.projection_variables.push_back(std::string(var.name()));
        }

        features.join_variables = sparql_query.get_join_variable_names();
        features.non_join_variables = sparql_query.get_non_join_variable_names();
        features.num_triple_patterns = sparql_query.triple_patterns_.size();
        features.is_distinct = sparql_query.distinct_;

        // Print basic query features
        std::cout << "Variable Names: ";
        for (const auto &name : features.variable_names) {
            std::cout << name << " ";
        }
        std::cout << "\nProjection Variables: ";
        for (const auto &proj : features.projection_variables) {
            std::cout << proj << " ";
        }
        std::cout << "\nJoin Variables: ";
        for (const auto &join : features.join_variables) {
            std::cout << join << " ";
        }
        std::cout << "\nNon-Join Variables: ";
        for (const auto &non_join : features.non_join_variables) {
            std::cout << non_join << " ";
        }
        std::cout << "\nNumber of Triple Patterns: " << features.num_triple_patterns << "\n";
        std::cout << "Is Distinct: " << (features.is_distinct ? "true" : "false") << "\n";


        // Extract ODG features
        auto& odg = const_cast<query::OperandDependencyGraph&>(sparql_query.odg_);
        features.num_connected_components = odg.union_components().size();
        auto operands = this->triplestore_.get_query_operands(this->triplestore_.get_hypertrie(),
                                                             sparql_query.get_slice_keys());
        // Create adjacency matrix for ODG
        size_t num_vars = sparql_query.var_to_id_.size();
        features.adjacency_matrix = std::vector<std::vector<int>>(num_vars, std::vector<int>(num_vars, 0));
        features.variable_degrees.resize(num_vars, 0);

        // Map variable IDs to indices
        std::unordered_map<char, size_t> var_id_to_index;
        size_t index = 0;
        for (const auto &[var, var_id] : sparql_query.var_to_id_) {
            var_id_to_index[var_id] = index++;
        }

        // Build adjacency matrix based on shared triple patterns
    for (size_t tp_idx = 0; tp_idx < sparql_query.triple_patterns_.size(); ++tp_idx) {
        std::vector<char> vars_in_tp;

        // Extract variables from this triple pattern
        for (const auto &[var, var_id] : sparql_query.var_to_id_) {
            // Check if variable appears in this triple pattern
            if (std::ranges::contains(sparql_query.triple_patterns_[tp_idx], var)) {
                vars_in_tp.push_back(var_id);
            }
        }

        // Connect all pairs of variables in the same triple pattern
        for (size_t i = 0; i < vars_in_tp.size(); ++i) {
            for (size_t j = i + 1; j < vars_in_tp.size(); ++j) {
                size_t idx1 = var_id_to_index[vars_in_tp[i]];
                size_t idx2 = var_id_to_index[vars_in_tp[j]];
                features.adjacency_matrix[idx1][idx2] = 1;
                features.adjacency_matrix[idx2][idx1] = 1;
            }
        }

        // Update degrees
        for (char var_id : vars_in_tp) {
            features.variable_degrees[var_id_to_index[var_id]]++;
        }
    }
        // Print graph features
        std::cout << "Adjacency Matrix:\n";
        for (const auto &row : features.adjacency_matrix) {
            for (int val : row) {
                std::cout << val << " ";
            }
            std::cout << "\n";
        }
        std::cout << "Variable Degrees: ";
        for (int degree : features.variable_degrees) {
            std::cout << degree << " ";
        }
        std::cout << "\n";

    // Calculate graph metrics
    int total_edges = 0;
    for (const auto &row : features.adjacency_matrix) {
        for (int val : row) {
            total_edges += val;
        }
    }
    total_edges /= 2; // Undirected graph

    features.max_variable_degree = *std::max_element(features.variable_degrees.begin(), features.variable_degrees.end());
    features.min_variable_degree = *std::min_element(features.variable_degrees.begin(), features.variable_degrees.end());
    features.avg_variable_degree = std::accumulate(features.variable_degrees.begin(), features.variable_degrees.end(), 0.0) / num_vars;
    features.graph_density = static_cast<double>(total_edges) / (num_vars * (num_vars - 1) / 2.0);

        std::cout << "Max Variable Degree: " << features.max_variable_degree << "\n";
        std::cout << "Min Variable Degree: " << features.min_variable_degree << "\n";
        std::cout << "Avg Variable Degree: " << features.avg_variable_degree << "\n";
        std::cout << "Graph Density: " << features.graph_density << "\n";


    // Calculate cardinality features
    using CardEst = query::operators::CardinalityEstimation<dice::rdf_tensor::htt_t, dice::rdf_tensor::allocator_type>;

    try {
        double min_cardinality = std::numeric_limits<double>::max();

        for (const auto &[var, var_id] : sparql_query.var_to_id_) {
            double card = CardEst::calcCard(odg, operands, var_id);
            features.variable_cardinalities.push_back(card);

            if (card < min_cardinality) {
                min_cardinality = card;
                features.min_cardinality_variable = var_id;
            }
        }

        std::cout << "Variable Cardinalities: ";
        for (double card : features.variable_cardinalities) {
            std::cout << card << " ";
        }
        std::cout << "\nMin Cardinality Variable: " << features.min_cardinality_variable << "\n";

        // Create rdf_tensor::Query for total estimate
        std::vector<char> proj_vars_id;
        for (auto const &proj_var : sparql_query.projected_variables_) {
            proj_vars_id.push_back(sparql_query.var_to_id_.at(proj_var));
        }
        rdf_tensor::Query rdf_query{odg, operands, proj_vars_id, timeout};

        features.total_query_cardinality = CardEst::estimate(odg, operands, rdf_query);

        std::cout << "Total Query Cardinality: " << features.total_query_cardinality << "\n";

    } catch (const std::exception& e) {
        spdlog::warn("Error calculating cardinality: {}", e.what());
    }

    return features;
    }

    // Callback function to write API response data
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
        ((std::string*)userp)->append((char*)contents, size * nmemb);
        return size * nmemb;
    }

    std::string send_query_features_to_api(const dice::endpoint::QueryFeatures& features) {
        CURL* curl = curl_easy_init();
        std::string response_data;

        if (curl) {
            CURLcode res;

            // Convert QueryFeatures to JSON
            std::string json_data = features.to_json();

            // Set the URL for the API endpoint
            curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:8000/query-features/");

            // Set the HTTP POST method
            curl_easy_setopt(curl, CURLOPT_POST, 1L);

            // Set the JSON data as the POST body
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data.c_str());

            // Set the Content-Type header to application/json
            struct curl_slist* headers = nullptr;
            headers = curl_slist_append(headers, "Content-Type: application/json");
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

            // Set up callback to capture response
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);

            // Perform the request
            res = curl_easy_perform(curl);

            // Check for errors
            if (res != CURLE_OK) {
                std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
            } else {
                std::cout << "Query features sent successfully!" << std::endl;
                std::cout << "Response: " << response_data << std::endl;
            }

            // Clean up
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
        } else {
            std::cerr << "Failed to initialize CURL." << std::endl;
        }

        return response_data;
    }

    // Send runtime in seconds to the /query-runtime/ endpoint
    void send_runtime_to_api(double runtime_seconds) {
        CURL* curl = curl_easy_init();
        std::string response_data;

        if (curl) {
            CURLcode res;
            // Prepare JSON payload
            std::string json_data = "{\"runtime\": " + std::to_string(runtime_seconds) + "}";

            curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:8000/query-runtime/");
            curl_easy_setopt(curl, CURLOPT_POST, 1L);
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data.c_str());

            struct curl_slist* headers = nullptr;
            headers = curl_slist_append(headers, "Content-Type: application/json");
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

            // Optional: capture response
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);

            res = curl_easy_perform(curl);

            if (res != CURLE_OK) {
                std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
            } else {
                std::cout << "Runtime sent successfully! Response: " << response_data << std::endl;
            }

            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
        } else {
            std::cerr << "Failed to initialize CURL." << std::endl;
        }
    }

    std::vector<std::string> parse_query_plan(const std::string &api_response) {
        nlohmann::json response_json = nlohmann::json::parse(api_response);
        return response_json["query_plan"].get<std::vector<std::string>>();
    }

    void SPARQLEndpoint::handle_query(restinio::request_handle_t req, std::chrono::steady_clock::time_point timeout) {
        using namespace dice::sparql2tensor;
        using namespace restinio;

        auto sparql_query = parse_sparql_query_param(req, this->sparql_query_cache_);
        if (not sparql_query)
            return;

        // Collect query features for DRL
        QueryFeatures features = extract_query_features(*sparql_query, timeout);

        // Method 3: Send to API immediately
        // Send query features to API and get the response
        std::string api_response = send_query_features_to_api(features);
        std::cout << "API Response: " << api_response << std::endl;
        std::vector<std::string> query_plan = parse_query_plan(api_response);
        // for (const std::string& s : query_plan) {
        //     std::cout << s << "\t";
        // }
        // std::cout << std::endl;

        auto start_time = std::chrono::steady_clock::now(); // Start timing

        if (sparql_query->ask_) {
            bool ask_res = this->triplestore_.eval_ask(*sparql_query, timeout);
            std::string res = ask_res ? "true" : "false";
            req->create_response(status_ok())
                    .append_header(http_field::content_type, "application/sparql-results+json")
                    .set_body(R"({ "head" : {}, "boolean" : )" + res + " }")
                    .done();
        } else {
            SparqlJsonResultSAXWriter json_writer{sparql_query->projected_variables_, 100'000};

            for (auto const &entry : this->triplestore_.eval_select(*sparql_query, timeout, query_plan)) {
                json_writer.add(entry);
            }
            json_writer.close();
            check_timeout(timeout);

            req->create_response(status_ok())
                    .append_header(http_field::content_type, "application/sparql-results+json")
                    .set_body(std::string{json_writer.string_view()})
                    .done();
            spdlog::info("HTTP response {}: {} variables, {} solutions, {} bindings",
                         status_ok(),
                         sparql_query->projected_variables_.size(),
                         json_writer.number_of_written_solutions(),
                         json_writer.number_of_written_bindings());
        }

        auto end_time = std::chrono::steady_clock::now(); // End timing
        auto execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        double runtime_seconds = execution_time / 1000.0;
        send_runtime_to_api(runtime_seconds);
        spdlog::info("Query execution time: {} ms", execution_time); // Log execution time


    }

}// namespace dice::endpoint