from src.models.feature_extraction_v2 import FeatureExtractionV2
from src.utils.file_handling import set_training_data

def extract_features_v2(data, filename, nlp, word_freq):
    extracted_features = []
    for item in data:
        response = {}
        level3 = {}
        response['question_text'] = item['question_text']
        response['answer'] = item['answer']
        response['option_A'] = item['options_A']
        response['option_B'] = item['options_B']
        response['option_C'] = item['options_C']
        response['option_D'] = item['options_D']
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
        level3['v1_options'] = opt_deps['v1_options']
        level3['v2_options'] = opt_deps['v2_options']
        level3['v3_options'] = opt_deps['v3_options']
        level3['v4_options'] = opt_deps['v4_options']
        level3['v5_options'] = opt_deps['v5_options']
        level3['v6_options'] = opt_deps['v6_options']
        level3['v7_options'] = opt_deps['v7_options']
        level3['v8_options'] = opt_deps['v8_options']
        level3['v9_options'] = opt_deps['v9_options']
        level3['v10_options'] = opt_deps['v10_options']
        level3['pro1_options'] = opt_deps['pro1_options']
        level3['pro2_options'] = opt_deps['pro2_options']
        level3['pro3_options'] = opt_deps['pro3_options']
        level3['n1_options'] = opt_deps['n1_options']
        level3['n2_options'] = opt_deps['n2_options']
        level3['n3_options'] = opt_deps['n3_options']
        level3['a1_options'] = opt_deps['a1_options']
        level3['a2_options'] = opt_deps['a2_options']
        level3['a3_options'] = opt_deps['a3_options']
        level3['a4_options'] = opt_deps['a4_options']
        level3['a5_options'] = opt_deps['a5_options']
        level3['a6_options'] = opt_deps['a6_options']
        level3['pre1_options'] = opt_deps['pre1_options']
        level3['pre2_options'] = opt_deps['pre2_options']
        level3['pre3_options'] = opt_deps['pre3_options']
        level3['con1_options'] = opt_deps['con1_options']
        level3['con2_options'] = opt_deps['con2_options']
        level3['scon1_options'] = opt_deps['scon1_options']
        level3['scon2_options'] = opt_deps['scon2_options']
        level3['sva'] = opt_deps['sva']
        level3['sva_options'] = opt_deps['sva_options']
        level3['c_count'] = opt_deps['c_count']
        level3['s_count'] = opt_deps['s_count']
        level3['u_word'] = opt_deps['u_word']
        response['level3'] = level3
        temp = check_level2(level3)
        # response['level2'] = temp[0]
        response['level2_with_options'] = temp
        response['levels'] = check_levels(level3)
        response['options_comp'] = check_comp(response['level2_with_options'], response['option_A'], response['option_B'], response['option_C'], response['option_D'])
        extracted_features.append(response)
    set_training_data(f'{filename}_features_v2', extracted_features)

    return extracted_features

def check_level2(data):
    
    # level2 = {
    #     "main_verbs": data.get("v1", False) or data.get("v2", False) or data.get("v3", False),
    #     "tense": data.get("v4", False) or data.get("v5", False),
    #     "infinitives": data.get("v6", False),
    #     "passives": data.get("v7", False) or data.get("v8", False),
    #     "have_+_participle": data.get("v9", False),
    #     "auxiliary_verbs": data.get("v10", False),
    #     "pronouns": data.get("pro1", False) or data.get("pro2", False) or data.get("pro3", False),
    #     "nouns": data.get("n1", False) or data.get("n2", False) or data.get("n3", False),
    #     "determiners": data.get("a1", False) or data.get("a2", False) or data.get("a5", False),
    #     "other_adjectives": data.get("a3", False) or data.get("a4", False),
    #     "prepositions": data.get("pre1", False) or data.get("pre2", False) or data.get("pre3", False),
    #     "conjunctions": data.get("con1", False) or data.get("con2", False) or data.get("scon1", False) or data.get("scon2", False),
    #     "subject_verb_agreement": data.get("sva", False),
    # }
    
    level2_with_opt = {
        "main_verbs": data.get("v1_options", "") or data.get("v2_options", "") or data.get("v3_options", ""),
        "tense": data.get("v4_options", "") or data.get("v5_options", ""),
        "infinitives": data.get("v6_options", ""),
        "passives": data.get("v7_options", "") or data.get("v8_options", ""),
        "have_+_participle": data.get("v9_options", ""),
        "auxiliary_verbs": data.get("v10_options", ""),
        "pronouns": data.get("pro1_options", "") or data.get("pro2_options", "") or data.get("pro3_options", ""),
        "nouns": data.get("n1_options", "") or data.get("n2_options", "") or data.get("n3_options", ""),
        "determiners": data.get("a1_options", "") or data.get("a2_options", "") or data.get("a5_options", ""),
        "other_adjectives": data.get("a3_options", "") or data.get("a4_options", ""),
        "prepositions": data.get("pre1_options", "") or data.get("pre2_options", "") or data.get("pre3_options", ""),
        "conjunctions": data.get("con1_options", "") or data.get("con2_options", "") or data.get("scon1_options", "") or data.get("scon2_options", ""),
        "subject_verb_agreement": data.get("sva_options", ""),
    }
    
    level2_with_opt_list = [[key, value] for key, value in level2_with_opt.items()]

    # return [level2, level2_with_opt_list]
    return level2_with_opt_list

def check_comp(opt, A, B, C, D):
    
    cek = opt[0]
    Another = []
    Bnother = []
    Cnother = []
    Dnother = []
    Another.append(A[0])
    Bnother.append(B[0])
    Cnother.append(C[0])
    Dnother.append(D[0])
    for item in opt:
        if item[1] != "":
            if(item[1] in A[0]):
                Another.append(item[0])
            if(item[1] in B[0]):
                Bnother.append(item[0])
            if(item[1] in C[0]):
                Cnother.append(item[0])
            if(item[1] in D[0]):
                Dnother.append(item[0])

    return Another, Bnother, Cnother, Dnother

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