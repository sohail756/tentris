#ifndef DICE_SPARQL_PARSEDSPARQL_HPP
#define DICE_SPARQL_PARSEDSPARQL_HPP

#include <rdf4cpp/rdf.hpp>

#include <dice/hypertrie.hpp>
#include <dice/rdf-tensor/HypertrieTrait.hpp>
#include <dice/rdf-tensor/RDFNodeHashes.hpp>
#include <dice/rdf-tensor/Query.hpp>

#include <robin_hood.h>

namespace dice::sparql2tensor {

	struct SPARQLQuery {
		dice::query::OperandDependencyGraph odg_;

		std::vector<rdf4cpp::rdf::query::Variable> projected_variables_;

		robin_hood::unordered_map<rdf4cpp::rdf::query::Variable, char, dice::hash::DiceHashwyhash<rdf4cpp::rdf::query::Variable>> var_to_id_;

		std::vector<rdf4cpp::rdf::query::TriplePattern> triple_patterns_;

		rdf4cpp::rdf::IRIFactory prefixes_;

		bool distinct_ = false;

		bool ask_ = false;

		bool project_all_variables_ = false;

		SPARQLQuery() = default;

		static SPARQLQuery parse(std::string const &sparql_query_str);

		SPARQLQuery(std::string const &sparql_query_str) : SPARQLQuery(SPARQLQuery::parse(sparql_query_str)) {}

		[[nodiscard]] bool is_distinct() const noexcept;

		std::vector<rdf_tensor::SliceKey> get_slice_keys() const;

		//the following variables and methods are used to handle variable IDs and names. these are used to find query features.
		// Add this as a mutable member variable
		mutable robin_hood::unordered_map<char, rdf4cpp::rdf::query::Variable> id_to_var_;
		mutable bool reverse_map_built_ = false;

		// Helper method to build reverse map
		void build_reverse_map() const {
			if (!reverse_map_built_) {
				id_to_var_.clear();
				for (const auto& [variable, id] : var_to_id_) {
					id_to_var_[id] = variable;
				}
				reverse_map_built_ = true;
			}
		}

		mutable std::vector<std::string> join_variable_names_;
		mutable std::vector<std::string> non_join_variable_names_;
		mutable bool variables_computed_ = false;

		void compute_variable_types() const {
			if (variables_computed_) return;

			build_reverse_map(); // Ensure reverse map is built

			std::unordered_map<char, int> variable_count;

			// Count occurrences of each variable in triple patterns
			for (const auto &triple : triple_patterns_) {
				for (const auto &component : {triple.subject(), triple.predicate(), triple.object()}) {
					if (component.is_variable()) {
						char var_id = var_to_id_.at(component.as_variable());
						variable_count[var_id]++;
					}
				}
			}

			// Separate join and non-join variables
			for (const auto &[var_id, count] : variable_count) {
				if (count > 1) {
					join_variable_names_.push_back(std::string(id_to_var_.at(var_id).name()));
				} else {
					non_join_variable_names_.push_back(std::string(id_to_var_.at(var_id).name()));
				}
			}

			variables_computed_ = true;
		}

		[[nodiscard]] const std::vector<std::string> &get_join_variable_names() const {
			compute_variable_types();
			return join_variable_names_;
		}

		[[nodiscard]] const std::vector<std::string> &get_non_join_variable_names() const {
			compute_variable_types();
			return non_join_variable_names_;
		}

	};

}// namespace dice::sparql2tensor

#endif//DICE_SPARQL_PARSEDSPARQL_HPP