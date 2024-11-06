# from CoreNLG.NlgTools import NlgTools
# from CoreNLG.PredefObjects import TextVar


# class NLGCore:
#     """
#     Class to handle NLGCore
#     """

#     def __init__(self) -> None:
#         self.nlg = NlgTools()

#     def generate_text(self, text_list):
#         """
#         Function to generate text using CoreNLG
#         """
#         text_rule = TextVar(
#             self.nlg.nlg_syn('Kamu harus belajar',
#                              'Untuk meningkatkan kemampuan bahasa inggrismu, maka kamu harus belajar '),
#             ', '.join([text.lower() for text in text_list[:-1]]) + ', dan ' + text_list[-1].lower() + '.',
#             self.nlg.nlg_syn('Semoga rekomendasi nya bisa menjadikan motivasi untuk meningkatkan kemampuanmu!.',
#                              'Semangat, semoga sukses pada assessment selanjutnya')
#         )

#         text = self.nlg.write_text(text_rule)
#         return text
