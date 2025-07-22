#ifndef TENTRIS_SPARQLENDPOINT_HPP
#define TENTRIS_SPARQLENDPOINT_HPP

#include <dice/endpoint/Endpoint.hpp>

namespace dice::endpoint {

	struct QueryFeatures {
		// Basic query info
		std::vector<std::string> variable_names;
		std::vector<std::string> projection_variables;
		std::vector<std::string> join_variables;
		std::vector<std::string> non_join_variables;
		size_t num_triple_patterns;
		bool is_distinct;

		// Cardinality features
		std::vector<double> variable_cardinalities;
		double total_query_cardinality;
		char min_cardinality_variable;

		// ODG features (numerical representation)
		std::vector<std::vector<int>> adjacency_matrix;  // Variable connectivity
		std::vector<int> variable_degrees;               // How many TPs each variable appears in
		int num_connected_components;                    // Graph connectivity
		double graph_density;                           // Edge density

		// Additional graph metrics
		int max_variable_degree;
		int min_variable_degree;
		double avg_variable_degree;

		// JSON serialization for API
		std::string to_json() const;
	};

	class SPARQLEndpoint final : public Endpoint {
	public:
		SPARQLEndpoint(tf::Executor &executor, triple_store::TripleStore &triplestore, SparqlQueryCache &sparql_query_cache, EndpointCfg const &endpoint_cfg);

	protected:
		void handle_query(restinio::request_handle_t req, std::chrono::steady_clock::time_point timeout) override;
	private:
		QueryFeatures extract_query_features(const sparql2tensor::SPARQLQuery &sparql_query,
										   std::chrono::steady_clock::time_point timeout);


	};

}// namespace dice::endpoint
#endif//TENTRIS_SPARQLENDPOINT_HPP