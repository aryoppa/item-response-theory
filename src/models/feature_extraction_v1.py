class FeatureExtractionV1:
    def __init__(self, data, nlp):
        self.data = data
        self.nlp = nlp
    
    def main_verbs(self, index):
        """
        Extract problems with Main Verbs.
        - Verbs that Require an Infinitive in the Complement (v1)
        - Verbs that Require an -ing Form in the Complement (v2)

        Args:
            
        Returns:
            v1, v2 indicators (boolean)
        """
        is_verb = False
        v1 = False
        v2 = False
        if self.data['pos_tag'][index][2].startswith('VB'):
            is_verb = True
        elif self.data['pos_tag'][index-1][2] == 'TO':
            v1 = True
        if is_verb:
            if self.data['pos_tag'][index+1][2] == 'TO':
                v1 = True
            if self.data['pos_tag'][index+1][2] == 'VBG':
                v2 = True
        return v1, v2

    def tense(self, index):
        """
        Extract problems with Tense.
        - Irregular Past Forms (v3)

        Args:
            
        Returns:
            v3 indicators (boolean)
        """
        if self.data['pos_tag'][index][2] == 'VBD':
            return True
        else:
            return False

    def option_tags(self):
        response = {}
        options = ['A', 'B', 'C', 'D']
        response['v1'] = False
        response['v2'] = False
        response['v3'] = False
        response['v5'] = False

        for opt in options:
            for index in self.data[opt]:
                if response['v1'] == False or response['v2'] == False:
                    v1, v2 = self.main_verbs(index[0])
                    if v1 == True:
                        response['v1'] = True
                    if v2 == True:
                        response['v2'] = True
                if response['v3'] == False:
                    response['v3'] = self.tense(index[0])
                if response['v5'] == False:
                    response['v5'] = self.infinitives(index[0])
            
        return response
    
    def subjunctives(self):
        doc = self.nlp(self.data['question_text'])
        options = ['A', 'B', 'C', 'D']

        for token in doc:
            if token.dep_ == "ccomp" or token.dep_ == "advcl":
                for opt in options:
                    for item in self.data[opt]:
                        if item[1] == token.text and token.morph.get('Mood') != ['Ind']:
                            return True
        return False
    
    def infinitives(self, index):
        doc = self.nlp(self.data['question_text'])
        i = 0
        for token in doc:
            if self.data['pos_tag'][index][2].startswith('VB') and i == index-1:
                if token.tag_ == 'TO':
                    return True
                else:
                    break
            elif self.data['pos_tag'][index][2] == 'TO' and i == index+1:
                if token.pos_ == 'VERB' and token.text == token.lemma_:
                    return True
                else:
                    break
            i += 1
        return False
    
    def passives(self):
        doc = self.nlp(self.data['question_text'])
        v6 = False
        v7 = False
        passive = ['auxpass', 'nsubjpass', 'csubjpass']
        for token in doc:
            if token.dep_ in passive:
                v6 = True
            if token.dep_ == 'ccomp':
                for child in token.children:
                    if child.text.lower() == 'it':
                        v7 = True
                        break
        return v6, v7

    def have_participle(self, token, opt_item):
        if token.pos_ == 'VERB':
            if token.text == opt_item[1]:
                for child in token.children:
                    if child.text.lower() == 'have':
                        return True
        return False

    def auxiliary(self, token, opt_item):
        if token.dep_.startswith('aux'):
            if token.text == opt_item[1]:
                return True
        elif token.pos_ == 'VERB':
            if token.text == opt_item[1]:
                for child in token.children:
                    if child.dep_.startswith('aux'):
                        return True
        return False
    
    def option_deps(self):
        doc = self.nlp(self.data['question_text'])
        options = ['A', 'B', 'C', 'D']
        response = {}
        response['v8'] = False
        response['v9'] = False
        response['pro1'] = False
        response['pro2'] = False
        response['n1'] = False
        response['n2'] = False
        response['n3'] = False
        response['a1'] = False
        response['a2'] = False
        response['a3'] = False
        response['a4'] = False
        response['pre1'] = False
        response['pre2'] = False
        response['con1'] = False
        response['adv1'] = False

        for token in doc:
            for opt in options:
                for item in self.data[opt]:
                    if response['v8'] == False:
                        response['v8'] = self.have_participle(token, item)
                    if response['v9'] == False:
                        response['v9'] = self.auxiliary(token, item)
                    if response['pro1'] == False:
                        response['pro1'] = self.object_pronouns(token, item)
                    if response['pro2'] == False:
                        response['pro2'] = self.relative_pronouns(token, item)
                    if response['n1'] == False:
                        response['n1'] = self.singular_plural(token, item)
                    if response['n2'] == False:
                        response['n2'] = self.infinitive_ing_subject(token, item)
                    if response['n3'] == False:
                        response['n3'] = self.nominal_that_clause(token, item)
                    if response['a1'] == False:
                        response['a1'] = self.noun_qualifying_phrases(token, item)
                    if response['a2'] == False:
                        response['a2'] = self.no_mean_not_any(token, item)
                    if response['a3'] == False:
                        response['a3'] = self.adjective_noun(token, item)
                    if response['a4'] == False:
                        response['a4'] = self.adjective_so(token, item)
                    if response['pre1'] == False:
                        response['pre1'] = self.prep_addition(token, item)
                    if response['pre2'] == False:
                        response['pre2'] = self.prep_cause(token, item)
                    if response['con1'] == False:
                        response['con1'] = self.future_result(token, item)
                    if response['adv1'] == False:
                        response['adv1'] = self.adverbs(token, item)
                    
        return response
    
    def object_pronouns(self, token, opt_item):
        if token.dep_ == 'pobj' and token.tag_.startswith('PRP'):
            for anc in token.ancestors:
                if anc.tag_ == 'IN' and (anc.text == opt_item[1] or token.text == opt_item[1]):
                    return True
        return False
    
    def relative_pronouns(self, token, opt_item):
        relative_pronouns_words = ["who", "whom", "whose", "which", "that", "where"]
        if token.dep_ == "relcl":
            if opt_item[1] in relative_pronouns_words:
                return True
        return False
    
    def singular_plural(self, token, opt_item):
        if token.pos_ == 'NOUN' and token.morph.get('Number') and token.text == opt_item[1]:
            return True
        return False
    
    def infinitive_ing_subject(self, token, opt_item):
        subject = ['csubj', 'csubjpass']
        if token.dep_ in subject and token.text == opt_item[1]:
            return True
        return False
    
    def nominal_that_clause(self, token, opt_item):
        if token.dep_ == 'mark' and token.text == opt_item[1] and token.text.lower() == 'that':
            return True
        return False

    def noun_qualifying_phrases(self, token, opt_item):
        if token.text.lower() == 'the':
            for anc in token.ancestors:
                if anc.pos_ == 'NOUN' and anc.text == opt_item[1] and anc.morph.get('Number') == ['Sing']:
                    return True
        return False
    
    def no_mean_not_any(self, token, opt_item):
        if token.tag_ == 'DT' and token.text.lower() == 'no':
            for anc in token.ancestors:
                if anc.text == opt_item[1] or token.text == opt_item[1]:
                    return True
        return False
    
    def adjective_noun(self, token, opt_item):
        if token.pos_ == 'NOUN' and token.dep_ == 'compound':
            for anc in token.ancestors:
                if token.text == opt_item[1] or anc.text == opt_item[1]:
                    return True
                else:
                    break
        return False
    
    def adjective_so(self, token, opt_item):
        if token.text.lower() == 'so' and token.dep_ == 'advmod':
            for anc in token.ancestors:
                if token.text == opt_item[1] or anc.text == opt_item[1]:
                    return True
                else:
                    break
        return False
    
    def prep_addition(self, token, opt_item):
        if token.text.lower() == 'besides' and token.dep_ == 'prep' and token.pos_ == 'SCONJ':
            for child in token.children:
                if token.text == opt_item[1] or child.text == opt_item[1]:
                    return True
        return False
    
    def prep_cause(self, token, opt_item):
        if token.text.lower() == 'because' and token.dep_ == 'mark':
            for anc in token.ancestors:
                if token.text == opt_item[1] or anc.text == opt_item[1]:
                    return True
        elif token.text.lower() == 'because' and token.dep_ == 'prep':
            for child in token.children:
                if token.text == opt_item[1] or child.text == opt_item[1]:
                    return True
        return False
    
    def future_result(self, token, opt_item):
        if token.text.lower() == 'when' and token.pos_ == 'SCONJ':
            for anc in token.ancestors:
                if token.text == opt_item[1] or anc.text == opt_item[1]:
                    return True
        return False
    
    def adverbs(self, token, opt_item):
        if token.pos_ == 'ADV':
            for anc in token.ancestors:
                if token.text == opt_item[1] or anc.text == opt_item[1]:
                    return True
            for child in token.children:
                if token.text == opt_item[1] or child.text == opt_item[1]:
                    return True
        return False

    def subject_verb_agreement(self):
        suggested_words = []
        suggested_words_aux = []

        doc = self.nlp(self.data['question_text'])
        for token in doc:
            if token.dep_ == "nsubj":
                for anc in token.ancestors:
                    if anc.pos_ == "VERB" and anc.text not in suggested_words:
                        suggested_words.append(anc.text)
                        if len(list(anc.children)) != 0:
                            for child in anc.children:
                                if child.pos_ == "AUX" and child.text not in suggested_words_aux:
                                    suggested_words_aux.append(child.text)
        return suggested_words, suggested_words_aux
    
    def check_sva(self):
        suggested_words, suggested_words_aux = self.subject_verb_agreement()
        options = ['A', 'B', 'C', 'D']
        sva = False
        
        for opt in options:
            for item in self.data[opt]:
                if item[1] in suggested_words or item[1] in suggested_words_aux:
                    sva = True
        return sva