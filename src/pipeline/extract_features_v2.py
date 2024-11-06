from src.models.feature_extraction_v2 import FeatureExtractionV2
from src.utils.file_handling import set_training_data

def extract_features_v2(data, filename, nlp, word_freq):
    extracted_features = []
    for item in data:
        response = {}
        level3 = {}

        response['question_text'] = item['question_text']
        featex_v2 = FeatureExtractionV2(item, nlp, word_freq)
        opt_deps = featex_v2.option_deps()
        level3['v1'] = opt_deps['v1']
        level3['v2'] = opt_deps['v2']
        level3['v3'] = opt_deps['v3']
        level3['v4'] = opt_deps['v4']
        level3['v5'] = opt_deps['v5']
        level3['v6'] = opt_deps['v6']
        level3['v7'] = opt_deps['v7']
        level3['v8'] = opt_deps['v8']
        level3['v9'] = opt_deps['v9']
        level3['v10'] = opt_deps['v10']
        level3['pro1'] = opt_deps['pro1']
        level3['pro2'] = opt_deps['pro2']
        level3['pro3'] = opt_deps['pro3']
        level3['n1'] = opt_deps['n1']
        level3['n2'] = opt_deps['n2']
        level3['n3'] = opt_deps['n3']
        level3['a1'] = opt_deps['a1']
        level3['a2'] = opt_deps['a2']
        level3['a3'] = opt_deps['a3']
        level3['a4'] = opt_deps['a4']
        level3['a5'] = opt_deps['a5']
        level3['a6'] = opt_deps['a6']
        level3['pre1'] = opt_deps['pre1']
        level3['pre2'] = opt_deps['pre2']
        level3['pre3'] = opt_deps['pre3']
        level3['con1'] = opt_deps['con1']
        level3['con2'] = opt_deps['con2']
        level3['scon1'] = opt_deps['scon1']
        level3['scon2'] = opt_deps['scon2']
        level3['c_count'] = opt_deps['c_count']
        level3['s_count'] = opt_deps['s_count']
        level3['u_word'] = opt_deps['u_word']
        level3['sva'] = opt_deps['sva']
        response['level3'] = level3
        response['level2'] = check_level2(level3)
        response['levels'] = check_levels(level3)
        extracted_features.append(response)
    set_training_data(f'{filename}_features_v2', extracted_features)

    return extracted_features

def check_level2(data):
    level2 = {
        "main_verbs": data.get("v1", False) or data.get("v2", False) or data.get("v3", False),
        "tense": data.get("v4", False) or data.get("v5", False),
        "infinitives": data.get("v6", False),
        "passives": data.get("v7", False) or data.get("v8", False),
        "have_+_participle": data.get("v9", False),
        "auxiliary_verbs": data.get("v10", False),
        "pronouns": data.get("pro1", False) or data.get("pro2", False) or data.get("pro3", False),
        "nouns": data.get("n1", False) or data.get("n2", False) or data.get("n3", False),
        "determiners": data.get("a1", False) or data.get("a2", False) or data.get("a5", False),
        "other_adjectives": data.get("a3", False) or data.get("a4", False),
        "prepositions": data.get("pre1", False) or data.get("pre2", False) or data.get("pre3", False),
        "conjunctions": data.get("con1", False) or data.get("con2", False) or data.get("scon1", False) or data.get("scon2", False),
        "subject_verb_agreement": data.get("sva", False),
    }
    return level2

def check_levels(data):
    
    if((data.get('c_count') > 2 and (data.get('u_word') > 3)) 
       or (data.get('s_count') > 1 and (data.get('u_word') > 2))
       or (data.get('c_count') > 4) 
       or (data.get('s_count') > 3) 
       or (data.get('u_word') > 5)):    
        levels = {
            "Level_3" : True,
            "Level_2" : False,
            "Level_1" : False,
        }
    elif((data.get('c_count') > 0 and (data.get('u_word') > 2)) 
         or (data.get('s_count') > 0 and (data.get('u_word') > 1))
       or (data.get('c_count') > 2) 
       or (data.get('s_count') > 1) 
       or (data.get('u_word') > 2)):
        levels = {
            "Level_3" : False,
            "Level_2" : True,
            "Level_1" : False,
        }
    else:
        levels = {
            "Level_3" : False,
            "Level_2" : False,
            "Level_1" : True,
        }

    return levels