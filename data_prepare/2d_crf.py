# def ls(s):
#     dic = {}
#     start, ans = 0, 0
#     left, right = 0, 0
#     for i, c in enumerate(s):
#         if c in dic:
#             start = max(start, dic[c] + 1)
#
#         dic[c] = i
#
#         if ans < i - start + 1:
#             ans = i - start + 1
#             left, right = start, i + 1
#
#     return ans, left, right
#
#
# if __name__ == "__main__":
#     s = 'yzzdtzehaha'
#     n, left, right = ls(s)
#     res = max(left, len(s)-right)
#     print(res)
def ls(inputfile):
    dic = {}
    # init
    for i in range(0, 99999, 50):
        if i not in dic:
            dic[i] = 0
    # readfile
    with open(inputfile, encoding='utf-8') as file:
        content = file.readlines()
    # opt data
    for line in content:
        linelist = line.split(',')
        for j in linelist:
            ij = int(j)
            print('ij', ij)
            key = int(ij / 50) * 50
            print('key', key)
            if key in dic:
                dic[key] += 1
    return dic


if __name__ == "__main__":
    dic = ls(inputfile='../input.txt')
    for key, value in dic.items():
        print("{key} {value}".format(key=str(key) + "-" + str(key + 49), value=value))
