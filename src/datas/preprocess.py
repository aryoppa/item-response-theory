from src.utils.file_handling import set_training_data

class DataPreprocess():
    def __init__(self, data, nlp, filename):
        self.data = data
        self.nlp = nlp
        self.filename = filename
    
    def pos_tag(self, text):
        doc = self.nlp(text)
        counter = 0
        pos_tag = []
        for token in doc:
            pos_tag.append([counter, token.text, token.tag_])
            counter += 1
        return pos_tag
    
    def get_underlines(self, data):
        pos_tag = data['pos_tag']
        options = ['A', 'B', 'C', 'D']
        index = 0
        data['temp'] = []
        data['answer'] = {}
        for opt in options:
            # print(data[opt])
            temp_underline, index = self.option_underline(pos_tag, data[opt], index)
            for i in range(len(data[opt])):
                data[opt][i][0] = temp_underline[i]
                if(opt == data['key_answer']):
                    data['temp'].append([data[opt][i][1]])
                # data['temp1'].append([data[opt][i][1]])
            # data['options'] = " ".join(item[0] for item in data['temp'])
        data['answer'] = " ".join(item[0] for item in data['temp'])
        return True

    def option_underline(self, question, option, index):
        underline = []
        check = 0
        while check != len(option):
            for word in option:
                while index != len(question):
                    if word[1] == question[index][1]:
                        underline.append(index)
                        check += 1
                        break
                    index += 1

        return underline, index
    
    def start(self):
        options = ['A', 'B', 'C', 'D']
        counter = 0
        for data in self.data:
            data['id'] = counter
            print(data['question_text'])
            data['pos_tag'] = self.pos_tag(data['question_text'])
            data['options_A'] = []
            data['options_B'] = []
            data['options_C'] = []
            data['options_D'] = []
            for opt in options:
                if opt == 'A':
                    data['options_A'].append(data[opt])
                if opt == 'B':
                    data['options_B'].append(data[opt])
                if opt == 'C':
                    data['options_C'].append(data[opt])
                if opt == 'D':
                    data['options_D'].append(data[opt])
                data[opt] = self.pos_tag(data[opt])
            self.get_underlines(data)
            counter += 1
        set_training_data(f'{self.filename}_preprocessed', self.data)
        return self.data
