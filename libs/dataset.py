class Dataset(object):
    def __init__(self, args):
        self.args = args

    def load_dict(self, dic):
        with open(dic, 'r', encoding='UTF-8', ) as d:
            lang_dict = d.readlines()
            for i in range(len(lang_dict)):
                lang_dict[i] = lang_dict[i].rstrip()
        chars_dict = {char: i for i, char in enumerate(lang_dict)}

        return chars_dict