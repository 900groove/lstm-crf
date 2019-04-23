from SPARQLWrapper import SPARQLWrapper
import pandas as pd


sparql = SPARQLWrapper(
    endpoint='http://ja.dbpedia.org/sparql',
    returnFormat='json')


def get_data(offset):
    sparql.setQuery("""
    prefix dbpedia-owl: <http://dbpedia.org/ontology/>
    select ?name ?abstract
    where {
        ?company a dbpedia-owl:Company.
        ?company rdfs:label ?name .
        ?company <http://dbpedia.org/ontology/abstract> ?abstract .
        }
    OFFSET """ + str(offset) + """
    LIMIT 10000
    """)

    results = sparql.query().convert()

    name = []
    text = []
    for i in range(len(results['results']['bindings'])):
        name.append(results['results']['bindings'][i]['name']['value'])
        text.append(results['results']['bindings'][i]['abstract']['value'])

    return pd.DataFrame({'name': name, 'text': text})


if __name__ == '__main__':
    df = pd.DataFrame(columns=['name', 'text'])
    for i in range(10000, 1000000, 10000):
        result_df = get_data(i)
        if len(result_df) > 1:
            df = pd.concat([df, result_df], axis=0)
        else:
            break

    df.to_csv('../data/data.csv', index=False)
