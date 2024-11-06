from src.models.feature_extraction_v1 import FeatureExtractionV1
from src.utils.file_handling import set_training_data
import spacy

def extract_features(data, filename, nlp):
    extracted_features = []
    for item in data:
        response = {}
        level3 = {}
        level2 = {}

        response['question_text'] = item['question_text']
        featex_v2 = FeatureExtractionV1(item, nlp)
        opt_tags = featex_v2.option_tags()
        level3['v1'] = opt_tags['v1']
        level3['v2'] = opt_tags['v2']
        level3['v3'] = opt_tags['v3']
        level3['v4'] = featex_v2.subjunctives()
        level3['v5'] = opt_tags['v5']
        level3['v6'], level3['v7'] = featex_v2.passives()
        opt_deps = featex_v2.option_deps()
        level3['v8'] = opt_deps['v8']
        level3['v9'] = opt_deps['v9']
        level3['pro1'] = opt_deps['pro1']
        level3['pro2'] = opt_deps['pro2']
        level3['n1'] = opt_deps['n1']
        level3['n2'] = opt_deps['n2']
        level3['n3'] = opt_deps['n3']
        level3['a1'] = opt_deps['a1']
        level3['a2'] = opt_deps['a2']
        level3['a3'] = opt_deps['a3']
        level3['a4'] = opt_deps['a4']
        level3['pre1'] = opt_deps['pre1']
        level3['pre2'] = opt_deps['pre2']
        level3['con1'] = opt_deps['con1']
        level3['adv1'] = opt_deps['adv1']
        level3['s1'] = featex_v2.check_sva()
        response['level3'] = level3
        response['level2'] = check_level2(level3)
        extracted_features.append(response)
    set_training_data(f'{filename}_features', extracted_features)

    return extracted_features

def check_level2(data):
    level2 = {
        "main_verbs": data.get("v1", False) or data.get("v2", False),
        "tense": data.get("v3", False),
        "subjunctives": data.get("v4", False),
        "infinitives": data.get("v5", False),
        "passives": data.get("v6", False) or data.get("v7", False),
        "have_+_participle": data.get("v8", False),
        "auxiliary_verbs": data.get("v9", False),
        "pronouns": data.get("pro1", False) or data.get("pro2", False),
        "nouns": data.get("n1", False) or data.get("n2", False) or data.get("n3", False),
        "determiners": data.get("a1", False) or data.get("a2", False),
        "other_adjectives": data.get("a3", False) or data.get("a4", False),
        "prepositions": data.get("pre1", False) or data.get("pre2", False),
        "conjunctions": data.get("con1", False),
        "adverbs": data.get("adv1", False),
        "sentences_and_clauses": data.get("s1", False),
    }
    return level2
