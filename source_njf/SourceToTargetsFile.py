import json


class SourceToTargetsFile:

    def __init__(self, pairs):
        self.pairs = pairs
        # self.__file_list = file_list

    # def get_pair_inds(self):
    #     return self.__pair_inds

    # def get_pair_names(self):
    #     return [(self.__file_list[p], self.__file_list[q]) for p, q in self.__pair_inds]

    # def get_files(self):
    #     return self.__file_list

    def get_only_target_indices(self):
        a = {}
        for pair in self.__pair_inds:
            a[pair[0]] = False
            if pair[1] not in a:
                a[pair[1]] = True
        return {ind: True for (ind, val) in a.items() if val}

    def write(self, fname):
        write(fname, self.get_pair_names())


def load(fname):
    with open(fname) as file:
        data = json.load(file)
        pairs = data['pairs']
    # if isinstance(pairs[0][0], int):
    #     return None
    # files = {}

    # def get_or_set(f):
    #     if f not in files:
    #         files[f] = len(files)
    #     return files[f]

    # new_pairs = []
    # for pair in pairs:
    #     for i in range(2):
    #         pair[i] = get_or_set(pair[i])
    #     new_pairs.append(pair)
    # ret_files = [None] * len(files)  # list of size of files, init to None's
    # for f in files:
        # ret_files[files[f]] = f
    return SourceToTargetsFile(pairs)


def write(path, pairs):
    data = {'pairs': pairs}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
