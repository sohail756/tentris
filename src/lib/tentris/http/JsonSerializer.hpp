#ifndef TENTRIS_JSONSERIALIZER_HPP
#define TENTRIS_JSONSERIALIZER_HPP

#include "tentris/http/TimeoutException.hpp"
#include "tentris/store/RDF/Term.hpp"
#include "tentris/store/SPARQL/Variable.hpp"
#include "tentris/store/TripleStore.hpp"
#include "tentris/util/HTTPUtils.hpp"
#include "tentris/tensor/BoolHypertrie.hpp"
#include "QueryResultState.hpp"


#include <iostream>


namespace tentris::http {
	namespace {
		using namespace tentris::tensor;
		using Variable = tentris::store::sparql::Variable;
		using namespace std::chrono;
		using TripleStore = tentris::store::TripleStore;
		using Status = tentris::http::ResultState;
		using namespace tentris::store::rdf;
	}; // namespace

	template<typename RESULT_TYPE>
	Status streamJSON(const std::vector<Variable> &vars, Einsum<RESULT_TYPE> &results,
	                  restinio::response_builder_t<restinio::chunked_output_t> &stream,
	                  const TripleStore &store, const system_clock::time_point &timeout) {
		using namespace std::string_literals;

		ulong result_count = 0;
		const ulong flush_result_count = 500;

		stream.append_chunk(R"({"head":{"vars":[)");
		bool firstTime = true;
		for (const auto &var : vars) {
			if (firstTime) {
				firstTime = false;
				stream.append_chunk("\"{}\""_format(var.name));
			} else {
				stream.append_chunk(",\"{}\""_format(var.name));
			}
		}
		stream.append_chunk(R"(]},"results":{"bindings":[)");
		bool firstResult = true;

		for (const EinsumEntry<RESULT_TYPE> &result : results) {
			std::stringstream json_result{};
			json_result << "{";
			bool firstKey = true;
			for (const auto[binding, var] : zip(result.key, vars)) {
				Term &term = *store.getTermIndex().inv().at(binding);

				if (firstKey) {
					firstKey = false;
				} else {
					json_result << ",";
				}

				json_result << "\"" << var.name << "\":{";

				const Term::NodeType &termType = term.type();
				switch (termType) {
					case Term::URI:
						json_result << R"("type":"uri")";
						break;
					case Term::BNode:
						json_result << R"("type":"bnode")";
						break;
					case Term::Literal:
						json_result << R"("type":"literal")";
						break;
				} // todo check default

				json_result << R"(,"value":")" << escapeJsonString(term.get_value());
				if (termType == Term::Literal) {
					const Literal &literal = static_cast<Literal &>(term);
					if (literal.hasType())
						json_result << R"(","datatype":")" << literal.getType();
					else if (literal.hasLang())
						json_result << R"(","xml:lang":")" << literal.getLang();
				}
				json_result << "\"}";
			}
			json_result << "}";

			std::string json_result_binding = json_result.str();
			for ([[maybe_unused]] const auto &c : range(result.value)) {
				result_count += vars.size();
				if (firstResult) {
					firstResult = false;
					stream.append_chunk(json_result_binding);
				} else {
					stream.append_chunk(",");
					stream.append_chunk(json_result_binding);
					// flush the content from time to time.
					if (result_count > flush_result_count) {
						if (system_clock::now() > timeout) {
							stream.append_chunk("]}}\n");
							stream.done();
							return Status::SERIALIZATION_TIMEOUT;
						} else {
							result_count = 0;
							stream.flush();
						}
					}
				}
			}
		}
		stream.append_chunk("]}}\n");
		stream.done();
		return Status::OK;
	}
} // namespace tentris::http

#endif // TENTRIS_JSONSERIALIZER_HPP
