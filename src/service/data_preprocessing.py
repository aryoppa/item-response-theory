from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

# from src.service.nlg_core import NLGCore


class DataPreProcessing:
    """
    Class for handling data preprocessing
    """

    def __init__(self) -> None:
        # self.obj_nlg = NLGCore()
        self

    # def recommend_materials(self, student_competencies, rules):
    #     """
    #     Generate recommendation materials
    #     """
    #     student_recommendations = {}

    #     # Define the mapping of competencies to materials
    #     competency_to_material = {
    #         "main_verbs": "materi 1",
    #         "tense": "materi 2",
    #         "infinitives": "materi 3",
    #         "passives": "materi 4",
    #         "have_+_participle": "materi 5",
    #         "auxiliary_verbs": "materi 6",
    #         "pronouns": "materi 7",
    #         "nouns": "materi 8",
    #         "determiners": "materi 9",
    #         "other_adjectives": "materi 10",
    #         "prepositions": "materi 11",
    #         "conjunctions": "materi 12",
    #         "subject_verb_agreement": "materi 13"
    #     }

    #     material_details = {
    #         "materi 1": [
    #             "Penggunaan kata kerja utama dalam kalimat",
    #             "Perbedaan antara kata kerja aksi dan kata kerja statis",
    #             "Bentuk kata kerja dalam tenses berbeda"],
    #         "materi 2": [
    #             "Present Simple dan Present Continuous",
    #             "Past Simple dan Past Continuous",
    #             "Future Simple dan Future Continuous",
    #             "Present Perfect dan Past Perfect",
    #             "Penggunaan tenses dalam konteks berbeda"],
    #         "materi 3": [
    #             "Penggunaan infinitive (to + verb) dalam kalimat",
    #             "Infinitive dengan dan tanpa 'to'",
    #             "Penggunaan infinitive setelah kata kerja tertentu"],
    #         "materi 4": [
    #             "Struktur kalimat pasif",
    #             "Perubahan dari kalimat aktif ke pasif",
    #             "Penggunaan pasif dalam berbagai tenses"],
    #         "materi 5": [
    #             "Penggunaan Present Perfect Tense",
    #             "Struktur kalimat Present Perfect",
    #             "Penggunaan Past Perfect Tense"],
    #         "materi 6": [
    #             "Penggunaan kata kerja bantu (do, does, did)",
    #             "Penggunaan modal verbs (can, could, may, might, must, etc.)",
    #             "Bentuk negatif dan pertanyaan menggunakan kata kerja bantu"],
    #         "materi 7": [
    #             "Penggunaan pronoun subjek (I, you, he, she, it, we, they)",
    #             "Penggunaan pronoun objek (me, you, him, her, it, us, them)",
    #             "Penggunaan possessive pronouns (my, your, his, her, its, our, their)"],
    #         "materi 8": [
    #             "Penggunaan kata benda dalam kalimat",
    #             "Singular dan plural nouns",
    #             "Countable dan uncountable nouns"],
    #         "materi 9": [
    #             "Penggunaan determiners (a, an, the)",
    #             "Penggunaan quantifiers (some, any, few, many, etc.)",
    #             "Penggunaan demonstrative determiners (this, that, these, those)"],
    #         "materi 10": [
    #             "Penggunaan adjective dalam kalimat",
    #             "Perbandingan adjective (comparative dan superlative)",
    #             "Penggunaan adjective dalam berbagai posisi dalam kalimat"],
    #         "materi 11": [
    #             "Penggunaan prepositions of place (in, on, at, etc.)",
    #             "Penggunaan prepositions of time (in, on, at, etc.)",
    #             "Prepositions setelah kata kerja tertentu (depend on, listen to, etc.)"],
    #         "materi 12": [
    #             "Penggunaan coordinating conjunctions (and, but, or, etc.)",
    #             "Penggunaan subordinating conjunctions (because, although, if, etc.)",
    #             "Penggunaan correlative conjunctions (either...or, neither...nor, etc.)"],
    #         "materi 13": [
    #             "Kesepakatan antara subjek dan kata kerja",
    #             "Penggunaan kata kerja dengan subjek tunggal dan jamak",
    #             "Kesepakatan dalam kalimat kompleks"]}

    #     # Iterate through each student's competencies
    #     for student_data in student_competencies:
    #         student_name = student_data["name"]
    #         competencies = set(student_data["competencies"])
    #         # Initialize an empty set to store recommended materials for each
    #         # student
    #         recommendations = set()

    #         # Iterate through each association rule
    #         for idx, rule in rules.iterrows():
    #             antecedents = set(rule['antecedents'])
    #             consequents = set(rule['consequents'])

    #             uncompeten = list(
    #                 set(competency_to_material.keys()) - competencies)

    #             # Check if the student is missing any antecedents
    #             missing_antecedents = antecedents - set(uncompeten)

    #             if missing_antecedents:
    #                 # Recommend all materials related to the missing
    #                 # antecedents
    #                 for antecedent in missing_antecedents:
    #                     if antecedent in competency_to_material:
    #                         recommendations.add(
    #                             competency_to_material[antecedent])

    #         # Map student to recommended materials with details
    #         student_material_details = []
    #         for material in recommendations:
    #             if material in material_details:
    #                 student_material_details.extend(material_details[material])

    #         student_recommendations[student_name] = self.obj_nlg.generate_text(
    #             student_material_details)

    #     return student_recommendations

    def transform_result_to_biner(self, test_result, questions):
        """
        This function is to transform result to biner data
        """
        question_list = []

        for i in range(len(questions)):
            question_list.append(f"soal {i+1}")
        
        index=0
        for q in question_list:
            for i in range(len(test_result)):
                if questions["key"][index] == "":
                    test_result.loc[i, q] = 0
                elif test_result.loc[i, q] == questions["key"][index]:
                    test_result.loc[i, q] = 1
                else:
                    test_result.loc[i, q] = 0
            index += 1
        
        return test_result

    def mapping_student_competency(
            self,
            transformed_data,
            df_mapping_question_comp):
        '''
        Implement to map student wrong answers with competencies.
        '''

        # Convert the first column (score) to a separate series and drop it
        student_answers = transformed_data.iloc[:, 1:]

        # Competencies list
        lib = [
            "main_verbs",
            "tense",
            "infinitives",
            "passives",
            "have_+_participle",
            "auxiliary_verbs",
            "pronouns",
            "nouns",
            "determiners",
            "other_adjectives",
            "prepositions",
            "conjunctions",
            "subject_verb_agreement"
        ]

        student_list = []

        for idx, row in student_answers.iterrows():
            student = {"name": f"student_{idx+1}", "competencies": set()}

            for question_index, answer in row.items():
                if answer == 0:  # If the answer is wrong
                    try:
                        # Strip any leading or trailing spaces
                        question_index = question_index.strip()
                        # Extract the question number
                        question_num = int(question_index.split(' ')[-1]) - 1
                        if question_num < len(df_mapping_question_comp):
                            for comp in lib:
                                if df_mapping_question_comp.at[question_num, comp]:
                                    student["competencies"].add(comp)
                    except (ValueError, IndexError) as e:
                        print(
                            f"Skipping invalid question index '{question_index}' for student {idx+1}: {e}")

            # Convert the set to a list for JSON serialization or other
            # processing
            student["competencies"] = list(student["competencies"])
            student_list.append(student)

        return student_list

    def generate_final_dataset(self, mapped_data):
        """
        This function can generate final dataset
        """
        final_dataset = []
        for element in mapped_data:
            final_dataset.append(element['competencies'])
        return final_dataset

    def data_transformation(self, final_dataset):
        """
        This function is to transform data after final data set was generated
        """
        tr = TransactionEncoder()
        tr_ary = tr.fit(final_dataset).transform(final_dataset)
        df_incorrect = pd.DataFrame(tr_ary, columns=tr.columns_)
        return df_incorrect
    
    # Convert association rules to JSON
    def association_rules_to_json(self, rules):
        rule_list = []
        for idx, rule in rules.iterrows():
            rule_dict = {
                "antecedents": list(rule['antecedents']),
                "consequents": list(rule['consequents']),
                "support": rule['support'],
                "confidence": rule['confidence'],
                "lift": rule['lift']
            }
            rule_list.append(rule_dict)
        return rule_list

    # # Convert association rules to JSON format
    # rules_json = association_rules_to_json(rules)
