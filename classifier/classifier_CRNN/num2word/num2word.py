import random
import re
from babel.numbers import format_currency
import decimal


class UnknownClassicalModeError(Exception):
    pass


class BadNumValueError(Exception):
    pass


class BadChunkingOptionError(Exception):
    pass


class NumOutOfRangeError(Exception):
    pass


class BadUserDefinedPatternError(Exception):
    pass


class BadRcFileError(Exception):
    pass


class BadGenderError(Exception):
    pass


STDOUT_ON = False


def print3(txt):
    if STDOUT_ON:
        print(txt)


# def enclose(s):
#     return "(?:%s)" % s


# def joinstem(cutpoint=0, words=""):
#     """
#     join stem of each word in words into a string for regex
#     each word is truncated at cutpoint
#     cutpoint is usually negative indicating the number of letters to remove
#     from the end of each word

#     e.g.
#     joinstem(-2, ["ephemeris", "iris", ".*itis"]) returns
#     (?:ephemer|ir|.*it)

#     """
#     return enclose("|".join(w[:cutpoint] for w in words))


# def bysize(words):
#     """
#     take a list of words and return a dict of sets sorted by word length
#     e.g.
#     ret[3]=set(['ant', 'cat', 'dog', 'pig'])
#     ret[4]=set(['frog', 'goat'])
#     ret[5]=set(['horse'])
#     ret[8]=set(['elephant'])
#     """
#     ret = {}
#     for w in words:
#         if len(w) not in ret:
#             ret[len(w)] = set()
#         ret[len(w)].add(w)
#     return ret


# def make_pl_si_lists(lst, plending, siendingsize, dojoinstem=True):

#     given a list of singular words: lst
#     an ending to append to make the plural: plending
#     the number of characters to remove from the singular
#         before appending plending: siendingsize
#     a flag whether to create a joinstem: dojoinstem

#     return:
#     a list of pluralised words: si_list (called si because this is what you need to
#                                          look for to make the singular)
#     the pluralised words as a dict of sets sorted by word length: si_bysize
#     the singular words as a dict of sets sorted by word length: pl_bysize
#     if dojoinstem is True: a regular expression that matches any of the stems: stem

#     if siendingsize is not None:
#         siendingsize = -siendingsize
#     si_list = [w[:siendingsize] + plending for w in lst]
#     pl_bysize = bysize(lst)
#     si_bysize = bysize(si_list)
#     if dojoinstem:
#         stem = joinstem(siendingsize, lst)
#         return si_list, si_bysize, pl_bysize, stem
#     else:
#         return si_list, si_bysize, pl_bysize


# ordinal_suff = "|".join(list(ordinal.keys()))


# NUMBERS

unit = ["", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
teen = [
    "mười",
    "mười một",
    "mười hai",
    "mười ba",
    "mười bốn",
    "mười năm",
    "mười sáu",
    "mười bảy",
    "mười tám",
    "mười chín",
]
ten = [
    "",
    "",
    "hai mươi",
    "ba mươi",
    "bốn mươi",
    "năm mươi",
    "sáu mươi",
    "bảy mươi",
    "tám mươi",
    "chín mươi",
]
# thousand = ['nghìn', "ngàn"]
mill = [
    " ",
    " nghìn",
    " triệu",
    " tỷ",
    " nghìn tỷ",
    " triệu tỷ",
    " tỷ tỷ",
    " nghìn tỷ tỷ",
    " triệu tỷ tỷ",
    " tỷ tỷ tỷ",
    " nghìn tỷ tỷ tỷ",
    " triệu tỷ tỷ tỷ",
]

# SUPPORT CLASSICAL PLURALIZATIONS

def_classical = dict(
    all=False, zero=False, herd=False, names=True, persons=False, ancient=False
)

all_classical = {k: True for k in list(def_classical.keys())}
no_classical = {k: False for k in list(def_classical.keys())}

# Maps strings to built-in constant types
string_to_constant = {"True": True, "False": False, "None": None}


class Words(str):
    def __init__(self, orig):
        self.lower = self.lower()
        self.split = self.split()
        self.first = self.split[0]
        self.last = self.split[-1]


class engine:
    def __init__(self):

        self.classical_dict = def_classical.copy()
        self.persistent_count = None
        self.mill_count = 0
        self.pl_sb_user_defined = []
        self.pl_v_user_defined = []
        self.pl_adj_user_defined = []
        self.si_sb_user_defined = []
        self.A_a_user_defined = []
        self.thegender = "neuter"

    deprecated_methods = dict(
        pl="plural",
        plnoun="plural_noun",
        plverb="plural_verb",
        pladj="plural_adj",
        sinoun="single_noun",
        prespart="present_participle",
        numwords="number_to_words",
        plequal="compare",
        plnounequal="compare_nouns",
        plverbequal="compare_verbs",
        pladjequal="compare_adjs",
        wordlist="join",
    )

    def __getattr__(self, meth):
        if meth in self.deprecated_methods:
            print3(
                "{}() deprecated, use {}()".format(meth, self.deprecated_methods[meth])
            )
            raise DeprecationWarning
        raise AttributeError

    # NUMERICAL INFLECTIONS

    # def ordinal(self, num):
    #     """
    #     Return the ordinal of num.
    #
    #     num can be an integer or text
    #
    #     e.g. ordinal(1) returns '1st'
    #     ordinal('one') returns 'first'
    #
    #     """
    #     if re.match(r"\d", str(num)):
    #         try:
    #             num % 2
    #             n = num
    #         except TypeError:
    #             if "." in str(num):
    #                 try:
    #                     # numbers after decimal,
    #                     # so only need last one for ordinal
    #                     n = int(num[-1])
    #
    #                 except ValueError:  # ends with '.', so need to use whole string
    #                     n = int(num[:-1])
    #             else:
    #                 n = int(num)
    #         try:
    #             post = nth[n % 100]
    #         except KeyError:
    #             post = nth[n % 10]
    #         return "{}{}".format(num, post)
    #     else:
    #         mo = re.search(r"(%s)\Z" % ordinal_suff, num)
    #         try:
    #             post = ordinal[mo.group(1)]
    #             return re.sub(r"(%s)\Z" % ordinal_suff, post, num)
    #         except AttributeError:
    #             return "%sth" % num

    def millfn(self, ind=0):
        if ind > len(mill) - 1:
            print3("number out of range")
            raise NumOutOfRangeError
        return mill[ind]

    def unitfn(self, units, mindex=0):
        return "{}{}".format(unit[units], self.millfn(mindex))

    def tenfn(self, tens, units, mindex=0):
        unit_str = ''
        if units == 1:
            unit_str = "mốt" if tens > 1 else unit[units]
        elif units == 4:
            unit_str = "tư" if tens > 1 else unit[units]
        elif units == 5:
            unit_str = "lăm" if tens > 0 else unit[units]
        else:
            unit_str = unit[units]
        if tens != 1:
            return "{}{}{}{}".format(
                ten[tens],
                " " if tens and units else "",
                unit_str,
                self.millfn(mindex),
            )
        return "{}{}".format(teen[units], mill[mindex])

    def hundfn(self, hundreds, tens, units, mindex):
        if hundreds:
            # andword = " %s " % self.number_args["andword"] if tens and units else ""
            # print (self.millfn(mindex))
            if tens == 0 and units:
                andword = " %s " % self.number_args["andword"]
            else:
                andword = " "
            return "{} trăm{}{}{}, ".format(
                unit[hundreds],  # use unit not unitfn as simpler
                andword,
                self.tenfn(tens, units),
                self.millfn(mindex),
            )
        if tens or units:
            return "{}{}, ".format(self.tenfn(tens, units), self.millfn(mindex))
        return ""

    def group1sub(self, mo):
        units = int(mo.group(1))
        if units == 1:
            return " %s, " % self.number_args["one"]
        elif units:
            return "%s, " % unit[units]
        else:
            return " %s, " % self.number_args["zero"]

    def group1bsub(self, mo):
        units = int(mo.group(1))
        if units:
            return "%s, " % unit[units]
        else:
            return " %s, " % self.number_args["zero"]

    def group2sub(self, mo):
        tens = int(mo.group(1))
        units = int(mo.group(2))
        if tens:
            return "%s, " % self.tenfn(tens, units)
        if units:
            return " {} {}, ".format(self.number_args["zero"], unit[units])
        return " {} {}, ".format(self.number_args["zero"], self.number_args["zero"])

    def group3sub(self, mo):
        hundreds = int(mo.group(1))
        tens = int(mo.group(2))
        units = int(mo.group(3))
        if hundreds == 1:
            hunword = " %s" % self.number_args["one"]
        elif hundreds:
            hunword = "%s" % unit[hundreds]
        else:
            hunword = " %s" % self.number_args["zero"]
        if tens:
            tenword = self.tenfn(tens, units)
        elif units:
            tenword = " {} {}".format(self.number_args["zero"], unit[units])
        else:
            tenword = " {} {}".format(
                self.number_args["zero"], self.number_args["zero"]
            )
        return "{} {}, ".format(hunword, tenword)

    def hundsub(self, mo):
        ret = self.hundfn(
            int(mo.group(1)), int(mo.group(2)), int(mo.group(3)), self.mill_count
        )
        self.mill_count += 1
        return ret

    def tensub(self, mo):
        return "%s, " % self.tenfn(int(mo.group(1)), int(mo.group(2)), self.mill_count)

    def unitsub(self, mo):
        return "%s, " % self.unitfn(int(mo.group(1)), self.mill_count)

    def enword(self, num, group, currency='vnd'):
        # import pdb
        # pdb.set_trace()

        if group == 1:
            num = re.sub(r"(\d)", self.group1sub, num)
        elif group == 2:
            num = re.sub(r"(\d)(\d)", self.group2sub, num)
            num = re.sub(r"(\d)", self.group1bsub, num, 1)
        elif group == 3:
            num = re.sub(r"(\d)(\d)(\d)", self.group3sub, num)
            num = re.sub(r"(\d)(\d)", self.group2sub, num, 1)
            num = re.sub(r"(\d)", self.group1sub, num, 1)
        elif int(num) == 0:
            num = self.number_args["zero"]
            # num = ''
        elif int(num) == 1:
            num = self.number_args["one"]
        else:
            num = num.lstrip().lstrip("0")
            self.mill_count = 0
            # surely there's a better way to do the next bit
            mo = re.search(r"(\d)(\d)(\d)(?=\D*\Z)", num)
            while mo:
                num = re.sub(r"(\d)(\d)(\d)(?=\D*\Z)", self.hundsub, num, 1)
                mo = re.search(r"(\d)(\d)(\d)(?=\D*\Z)", num)
            num = re.sub(r"(\d)(\d)(?=\D*\Z)", self.tensub, num, 1)
            num = re.sub(r"(\d)(?=\D*\Z)", self.unitsub, num, 1)
        return num

    def blankfn(self, mo):
        """ do a global blank replace
        TODO: surely this can be done with an option to re.sub
              rather than this fn
        """
        return ""

    def commafn(self, mo):
        """ do a global ',' replace
        TODO: surely this can be done with an option to re.sub
              rather than this fn
        """
        return ","

    def spacefn(self, mo):
        """ do a global ' ' replace
        TODO: surely this can be done with an option to re.sub
              rather than this fn
        """
        return " "

    def number_to_words(
            self,
            num,
            wantlist=False,
            group=0,
            comma="",
            andword="linh",
            currency=None,
            zero="không",
            one="một",
            decimal="và",
            threshold=None,
    ):
        """
        Return a number in words.

        group = 1, 2 or 3 to group numbers before turning into words
        comma: define comma
        andword: word for 'and'. Can be set to ''.
            e.g. "one hundred and one" vs "one hundred one"
        zero: word for '0'
        one: word for '1'
        decimal: word for decimal point
        threshold: numbers above threshold not turned into words

        parameters not remembered from last call. Departure from Perl version.
        """
        if currency is None:
            currency = ['đô la Mỹ', 'xu']
        self.number_args = dict(andword=andword, zero=zero, one=one)
        # print(self.number_args)
        num = "%s" % num

        # Handle "stylistic" conversions (up to a given threshold)...
        if threshold is not None and float(num) > threshold:
            spnum = num.split(".", 1)
            while comma:
                (spnum[0], n) = re.subn(r"(\d)(\d{3}(?:,|\Z))", r"\1,\2", spnum[0])
                if n == 0:
                    break
            try:
                return "{}.{}".format(spnum[0], spnum[1])
            except IndexError:
                return "%s" % spnum[0]

        if group < 0 or group > 3:
            raise BadChunkingOptionError
        nowhite = num.lstrip()
        if nowhite[0] == "+":
            sign = "dương"
        elif nowhite[0] == "-":
            sign = "âm"
        else:
            sign = ""

        # myord = num[-2:] in ("st", "nd", "rd", "th")
        # if myord:
        #     num = num[:-2]
        # print('num', num)
        finalpoint = False
        if decimal:
            if group != 0:
                chunks = num.split(comma)
            else:
                chunks = num.split(comma, 1)
            # print('chunks', chunks)
            if chunks[-1] == "":  # remove blank string if nothing after decimal
                chunks = chunks[:-1]
                finalpoint = True  # add 'point' to end of output
            elif len(chunks) > 1 and int(chunks[-1]) % 100 == 0:
                # print(int(chunks[-1])/100)
                chunks[-1] = str(int(int(chunks[-1]) / 100))

        else:
            chunks = [num]
        # print('chunks', chunks)

        first = 1
        loopstart = 0

        if chunks[0] == "":
            first = 0
            if len(chunks) > 1:
                loopstart = 1

        for i in range(loopstart, len(chunks)):
            chunk = chunks[i]
            # remove all non numeric \D
            chunk = re.sub(r"\D", self.blankfn, chunk)
            if chunk == "":
                chunk = "0"
            # print('ads',chunk)
            if group == 0 and (first == 0 or first == ""):
                chunk = self.enword(chunk, 0)
                # print('ads0',chunk)
            else:
                # print('ads1',chunk)
                chunk = self.enword(chunk, group)
            # print('ads',chunk)
            if chunk[-2:] == ", ":
                chunk = chunk[:-2]
            if group == 0 and first:
                chunk = re.sub(r", (\S+)\s+\Z", " %s \\1" % andword, chunk)
            # print(chunk)
            chunk = re.sub(r"\s+", self.spacefn, chunk)

            # chunk = re.sub(r"(\A\s|\s\Z)", self.blankfn, chunk)
            chunk = chunk.strip()
            if first:
                first = ""
            # print(chunk)
            chunks[i] = chunk
        # print('chunks', chunks)
        numchunks = []
        if first != 0:
            numchunks = chunks[0].split("%s " % comma)
        if len(currency) > 0:
            numchunks.append(currency[0])
        if chunks[-1] == "":  # remove blank string if nothing after decimal
            chunks = chunks[:-1]
        # print('chunks[1:]', chunks[1:])
        for chunk in chunks[1:]:
            numchunks.append(decimal)
            numchunks.extend(chunk.split("%s " % comma))
        if finalpoint:
            numchunks.append(decimal)
        if decimal in numchunks:
            if len(currency) > 1:
                numchunks.append(currency[1])
            elif len(currency) == 1:
                numchunks.append(currency[0])
        # print('numchunks', numchunks)
        # print('group', group)
        # wantlist: Perl list context. can explictly specify in Python
        if wantlist:
            if sign:
                numchunks = [sign] + numchunks
            return numchunks
        elif group:
            signout = "%s " % sign if sign else ""
            return "{}{}".format(signout, ", ".join(numchunks))
        else:
            signout = "%s " % sign if sign else ""
            num = "{}{}".format(signout, numchunks.pop(0))
            # print('num', num)
            if decimal is None:
                first = True
            else:
                first = not num.endswith(decimal)
            for nc in numchunks:
                if nc == decimal:
                    num += " %s" % nc
                    if len(currency) == 0:
                        first = 0
                elif nc == currency[0]:
                    num += " %s" % nc
                    first = 0
                elif first:
                    num += "{} {}".format(comma, nc)
                else:
                    num += " %s" % nc
                # print('num', num)
            return num

    # Join words with commas and a trailing 'and' (when appropriate)...

    def join(
            self,
            words,
            sep=None,
            sep_spaced=True,
            final_sep=None,
            conj="and",
            conj_spaced=True,
    ):
        """
        Join words into a list.

        e.g. join(['ant', 'bee', 'fly']) returns 'ant, bee, and fly'

        options:
        conj: replacement for 'and'
        sep: separator. default ',', unless ',' is in the list then ';'
        final_sep: final separator. default ',', unless ',' is in the list then ';'
        conj_spaced: boolean. Should conj have spaces around it

        """
        if not words:
            return ""
        if len(words) == 1:
            return words[0]

        if conj_spaced:
            if conj == "":
                conj = " "
            else:
                conj = " %s " % conj

        if len(words) == 2:
            return "{}{}{}".format(words[0], conj, words[1])

        if sep is None:
            if "," in "".join(words):
                sep = ";"
            else:
                sep = ","
        if final_sep is None:
            final_sep = sep

        final_sep = "{}{}".format(final_sep, conj)

        if sep_spaced:
            sep += " "

        return "{}{}{}".format(sep.join(words[0:-1]), final_sep, words[-1])


def money_to_words(
        num,
        comma='.',
        sep=2,
        currency="usd"):
    '''
    num = string number
    comma = , or .
    sep = 0|1|2:
        0: output without comma
        1: output within comma
        2: random comma for output
    currency = vnd or usd
    '''
    words = ""
    thousand = ['nghìn', 'ngàn']
    e = engine()
    if currency == 'usd':
        currency = [['đô la Mỹ', 'xu'], ['Đô la Mỹ', 'cents']]
        currency = random.choice(currency)
        words = e.number_to_words(num, comma=comma, andword="linh", decimal="và",
                                  currency=currency, )
    elif currency == 'vnd':

        currency = [['Việt Nam Đồng', ''], ['VND', ''], ['đồng', '']]
        currency = random.choice(currency)
        words = e.number_to_words(num, comma=comma, andword="linh", decimal="và",
                                  currency=currency, )
    if sep == 0:
        words = words.replace(',', '')
    elif sep == 2:
        com = [',', '']
        com = random.choice(com)
        words = words.replace(',', com)
    thousand = random.choice(thousand)
    words = words.replace('nghìn', thousand)
    return words


def generate_corpus_money(rrange=1000):
    lang_code = ['en_US', 'vi_VN']
    currency = ['VND', 'VNĐ', '$', '₫', '€', '¥', '£', '₩', 'USD', 'KRW']
    trans_currency = ['VND/USD', 'VNĐ/USD']
    exchange_corpus = []
    corpus_raw = []
    corpus = []
    cp_float_random_smallest = []
    cp_int_random_smallest = []
    for i in range(1, 999):
        corpus_raw.append('%.2f' % (random.uniform(0, 999)))
        corpus_raw.append('%d' % i)

    for i in range(1, rrange):
        exchange_corpus.append('%d' % (random.randint(17000, 30000)))
    for i in range(rrange):
        corpus_raw.append('%.2f' % (random.uniform(0, 999) * 1000))
        corpus_raw.append('%d' % (random.randint(0, 999) * 1000))
        corpus_raw.append('%d' % (random.randint(0, 999999)))

        corpus_raw.append('%.2f' % (random.uniform(0, 999999) * 1000))
        corpus_raw.append('%d' % (random.randint(0, 999) * 1000000))
        corpus_raw.append('%d' % (random.randint(0, 999999999)))

        corpus_raw.append('%.2f' % (random.uniform(0, 999999) * 1000000))
        corpus_raw.append('%d' % (random.randint(0, 999999) * 1000000))
        corpus_raw.append('%d' % (random.randint(0, 999999999999)))

    for w in exchange_corpus:
        # for loc in lang_code:
        loc = random.choice(lang_code)
        if '.' in w:
            vn = format_currency(w, '', locale=loc)
        else:
            vn = format_currency(w, '', locale=loc, format=u'#,##0')
        if random.choice([1, 10]) >= 7:
            vn = vn.replace(',', '').replace('.', '')
            # print(vn)
        c = random.choice(trans_currency)
        corpus.append(random.choice([vn.lstrip().rstrip() + c, vn.lstrip().rstrip() + ' ' + c]))

    for w in corpus_raw:
        # for loc in lang_code:
        loc = random.choice(lang_code)
        if '.' in w:
            vn = format_currency(w, '', locale=loc)
        else:
            vn = format_currency(w, '', locale=loc, format=u'#,##0')
        c = random.choice(currency)
        corpus.append(random.choice([vn.lstrip().rstrip() + ' ' + c]))

    # print(len(corpus))
    return corpus, corpus_raw


def generate_money_in_words(number_corpus):
    corpus = []
    currency = ['vnd', 'usd']
    for w in number_corpus:
        cur = random.choice(currency)
        if cur == 'vnd':
            vn = format_currency(w, '', locale='vi_VN', format=u'#0')
            words = money_to_words(vn, currency=cur)

        else:
            words = money_to_words(w, currency=cur)
        if random.choice([True, False]):
            words = words[0].upper() + words[1:]
        corpus.append(words)
    # print(corpus)
    return corpus


def generate_corpus():
    corpus_so, corpus_raw = generate_corpus_money(rrange=100)
    corpus_chu = generate_money_in_words(corpus_raw)
    random.shuffle(corpus_so)
    random.shuffle(corpus_chu)

    str_so = '\n'.join(corpus_so)
    str_chu = '\n'.join(corpus_chu)
    print(str_chu)
    # with open("so.txt", "w", encoding='utf-8') as f:
    #     f.write(str_so)
    # with open("chu.txt", "w", encoding='utf-8') as f:
    #     f.write(str_chu)


vn_number_system = {
    'không': 0,
    'mốt': 1,
    'một': 1,
    'hai': 2,
    'ba': 3,
    'bốn': 4,
    'tư': 4,
    'năm': 5,
    'lăm': 5,
    'sáu': 6,
    'bảy': 7,
    'tám': 8,
    'chín': 9,
    'mười': 10,
    'mười một': 11,
    'mười hai': 12,
    'mười ba': 13,
    'mười bốn': 14,
    'mười năm': 15,
    'mười sáu': 16,
    'mười bảy': 17,
    'mười tám': 18,
    'mười chín': 19,
    'hai mươi': 20,
    'ba mươi': 30,
    'bốn mươi': 40,
    'năm mươi': 50,
    'sáu mươi': 60,
    'bảy mươi': 70,
    'tám mươi': 80,
    'chín mươi': 90,
    'trăm': 100,
}

point = ['phảy', 'và']

vn_decimal_words = ['không', 'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín']
vn_teen = [
    "mười",
    "mười một",
    "mười hai",
    "mười ba",
    "mười bốn",
    "mười năm",
    "mười sáu",
    "mười bảy",
    "mười tám",
    "mười chín",
]
vn_ten = [
    "hai mươi",
    "ba mươi",
    "bốn mươi",
    "năm mươi",
    "sáu mươi",
    "bảy mươi",
    "tám mươi",
    "chín mươi",
]
vn_hund = ['trăm']
vn_thousand = ['nghìn', "ngàn"]
vn_mill = [
    'nghìn', "ngàn",
    "triệu",
    "tỷ",
    "nghìn tỷ",
    "ngàn tỷ",
    "triệu tỷ",
    "tỷ tỷ",
    "nghìn tỷ tỷ",
    "ngàn tỷ tỷ",
    "triệu tỷ tỷ",
    "tỷ tỷ tỷ",
    "nghìn tỷ tỷ tỷ",
    "ngàn tỷ tỷ tỷ",
    "triệu tỷ tỷ tỷ",
]
vn_mill.reverse()

vn_mill_dict = {
    'unit': 1,
    'nghìn': 1000,
    "ngàn": 1000,
    "triệu": 1000000,
    "tỷ": 1000000000,
    "nghìn tỷ": 1000000000000,
    "ngàn tỷ": 1000000000000,
    "triệu tỷ": 1000000000000000,
    "tỷ tỷ": 1000000000000000000,
    "nghìn tỷ tỷ": 1000000000000000000000,
    "ngàn tỷ tỷ": 1000000000000000000000,
    "triệu tỷ tỷ": 1000000000000000000000000,
    "tỷ tỷ tỷ": 1000000000000000000000000000,
}

curencyList = [
    "đô la mỹ",
    "việt nam đồng",
    "vnd",
    "usd",
    "$",
    "krw",
    "cents",
    "xu",
    "đồng",
    "đô",
    "la",
    "mỹ"
]

numberWord = [
    'không',
    'một',
    'mốt',
    'hai',
    'ba',
    'bốn',
    'tư',
    'năm',
    'lăm',
    'sáu',
    'bảy',
    'tám',
    'chín',
    'trăm',
    'nghìn',
    "ngàn",
    "triệu",
    "tỷ",
    'tỉ',
    'mười',
    'mươi',
    'phảy',
    'và',
    'linh',
    'lẻ',
    "đô la mỹ",
    "việt nam đồng",
    "vnd",
    "usd",
    "$",
    "krw",
    "cents",
    "xu",
    "đồng",
    "đô",
    "la",
    "mỹ"
]

odd = ['linh', 'lẻ']

vn_teen.reverse()


def number_formation(number_words, debug=False):
    num = 0
    for th in vn_thousand:
        group_thou = re.search(th, number_words, re.IGNORECASE)
        if group_thou is not None:
            input = number_words.strip().split(th)
            thou = input[0].lstrip().rstrip()
            num = num + vn_number_system[thou] * 1000
            number_words = input[1]
    group_hun = re.search('trăm', number_words, re.IGNORECASE)
    if group_hun is not None:
        input = number_words.strip().split('trăm')
        hund = input[0].lstrip().rstrip()
        num = num + vn_number_system[hund] * 100
        number_words = input[1]
    if number_words == '':
        return num
    else:
        for t in vn_ten:
            group_ten = re.search(t, number_words, re.IGNORECASE)
            if group_ten is not None:
                input = number_words.strip().split(t)
                if debug:
                    print(t)
                num = num + vn_number_system[t]
                number_words = input[1].rstrip().lstrip()
                if number_words != '':
                    num = num + vn_number_system[number_words]
                    number_words = ''

        for t in vn_teen:
            group_ten = re.search(t, number_words, re.IGNORECASE)
            if group_ten is not None:
                if debug:
                    print(t)
                input = number_words.strip().split(t)
                num = num + vn_number_system[t]
                number_words = input[1].rstrip().lstrip()
                if number_words != '':
                    num = num + vn_number_system[number_words]
        for o in odd:
            group_unit = re.search(o, number_words, re.IGNORECASE)
            if group_unit is not None:
                input = number_words.strip().split(o)
                _unit = input[1].lstrip().rstrip()
                num = num + vn_number_system[_unit]
                number_words = ''
        if debug:
            print('number_words', number_words)
        number_words = number_words.lstrip().rstrip()
        if ' ' in number_words:
            input = number_words.split(' ')
            num += vn_number_system[input[0]] * 10 + vn_number_system[input[1]]
        else:
            for u in vn_decimal_words:
                group_ten = re.search(u, number_words, re.IGNORECASE)
                if group_ten is not None:
                    num += vn_number_system[u]

    return num


def recursive_formation(intFraction, sum, num_key, debug=False):
    intFractionDict = {}
    if intFraction != '':
        intFractionDict = gex(intFraction, intFractionDict, 0)
    if debug:
        print('intFractionDict:', intFractionDict, '--sum:', sum, '--num_key:', num_key)
    sum_temp = 0
    for key in intFractionDict:
        strg = intFractionDict[key].replace(',', '')
        if key != 'tỷ':
            sum_temp += number_formation(strg, debug=debug) * vn_mill_dict[key]
        else:
            sum_temp = recursive_formation(strg, sum_temp, key)
    sum += sum_temp * vn_mill_dict[num_key]

    return sum


def fraction_formation(number_words):
    num = 0
    group_hun = re.search('trăm', number_words, re.IGNORECASE)
    if group_hun is not None:
        input = number_words.strip().split('trăm')
        hund = input[0].lstrip().rstrip()
        num = num + vn_number_system[hund] * 100
        number_words = input[1]
    if number_words == '':
        return num
    else:
        for t in vn_ten:
            group_ten = re.search(t, number_words, re.IGNORECASE)
            if group_ten is not None:
                input = number_words.strip().split(t)
                num = num + vn_number_system[t]
                number_words = input[1].rstrip().lstrip()
                if number_words != '':
                    num = num + vn_number_system[number_words]
                    number_words = ''

        for t in vn_teen:
            group_ten = re.search(t, number_words, re.IGNORECASE)
            if group_ten is not None:
                input = number_words.strip().split(t)
                num = num + vn_number_system[t]
                number_words = input[1].rstrip().lstrip()
                if number_words != '':
                    num = num + vn_number_system[number_words]
        group_unit = re.search('linh', number_words, re.IGNORECASE)
        if group_unit is not None:
            input = number_words.strip().split('linh')
            _unit = input[1].lstrip().rstrip()
            num = num + vn_number_system[_unit]
            number_words = ''
        for u in vn_decimal_words:
            group_ten = re.search(u, number_words, re.IGNORECASE)
            if group_ten is not None:
                num += vn_number_system[u]

    return num


def gex(input, dic, i):
    w = vn_mill[i]
    group = re.search(w, input, re.IGNORECASE)
    if group is not None:
        input = input.strip().split(w)
        dic[w] = input[0].lstrip().rstrip()
        input = input[1]
    i += 1
    if i < len(vn_mill):
        gex(input, dic, i)
    else:
        dic['unit'] = input
    return dic


def word_to_num(number_sentence, debug=False):
    if type(number_sentence) != str:
        raise ValueError(
            "Type of input is not string!")
    number_sentence = number_sentence.replace('-', ' ').replace(',', '')
    number_sentence = number_sentence.lower()  # converting input to lowercase
    number_sentence = number_sentence.replace('tỉ', 'tỷ')

    # filter all unnumber words
    temp_number_sentence = number_sentence
    for w in numberWord:
        group = re.search(w, temp_number_sentence, re.IGNORECASE)
        if group is not None:
            temp_number_sentence = temp_number_sentence.replace(group.group(), '')
    for i in range(len(temp_number_sentence)):
        temp_number_sentence = temp_number_sentence.replace('  ', ' ')

    temp_number_sentence = temp_number_sentence.split()
    for w in temp_number_sentence:
        number_sentence = number_sentence.replace(w, '')

    for i in range(len(number_sentence)):
        number_sentence = number_sentence.replace('  ', ' ')

    # print(number_sentence)
    existNumber = False
    for n in vn_decimal_words:
        group = re.search(n, number_sentence, re.IGNORECASE)
        if group is not None:
            existNumber = True
            break
    if not existNumber:
        return ''
    for p in curencyList:
        group = re.search(p, number_sentence, re.IGNORECASE)
        if group is not None:
            number_sentence = number_sentence.replace(p, '')  # strip extra spaces and split sentence into words

    if number_sentence.isdigit():  # return the number if user enters a number string
        return int(number_sentence)
    split_words = [number_sentence]
    for p in point:
        group = re.search(p, number_sentence, re.IGNORECASE)
        if group is not None:
            split_words = number_sentence.strip().split(p)
    decimalFraction = None
    intFraction = ''
    if len(split_words) > 1:
        decimal_avai = True
    else:
        decimal_avai = False
    if decimal_avai:
        decimalFraction = split_words[1]
        intFraction = split_words[0]
    else:
        intFraction = split_words[0]
    num_int = recursive_formation(intFraction, sum=0, num_key='unit', debug=debug)
    if decimalFraction:
        decimalFractionDict = {}
        if decimalFraction != '':
            decimalFractionDict = gex(decimalFraction, decimalFractionDict, 0)
        num_decimal = 0
        for key in decimalFractionDict:
            strg = decimalFractionDict[key].replace(',', '')
            num_decimal += fraction_formation(strg) * vn_mill_dict[key]
        if num_decimal < 10:
            num = str(num_int) + '.0' + str(num_decimal)
        else:
            num = str(num_int) + '.' + str(num_decimal)
        return num
    num = str(num_int)
    return num


def get_money_fr_str(string):
    currency = ''
    pos_currency = ''
    number = None
    amounts = re.findall(r'[\d\.-]+[.|,\d][\d\.-]+', string)
    if amounts:
        # print(amounts)
        for n in amounts:
            # strg = string.replace(',', '')
            input = string.strip().split(n)
            if input[0] == '':
                currency = input[1].strip()
                pos_currency = 'r'
            elif input[1] == '':
                currency = input[0].strip()
                pos_currency = 'l'
            # print(currency)
            number = n

    # print(currency, pos_currency, number)
    return currency, pos_currency, number


def test_word_to_num(r=10):
    currency = ['vnd', 'usd']
    for i in range(0, r):
        # print(i)
        n = random.uniform(0, 1000)
        n_str = '%.2f' % n
        cur = random.choice(currency)
        if cur == 'vnd':
            n_str = format_currency(n_str, '', locale='vi_VN', format=u'#0')
            words = money_to_words(n_str, currency=cur)

        else:
            words = money_to_words(n_str, currency=cur)
        if random.choice([True, False]):
            words = words[0].upper() + words[1:]
        num = word_to_num(words)
        # if float(num) != float(n_str):
        print(n, n_str, num, words)


if __name__ == '__main__':
    # abc = '57.260 VND, 1.237.123,23 USD, 1.232.112.322.323,99 USD, $99212370.86, 1234 VND'
    # abc = '57.260 VND'
    # n_str = format_currency('1232112322323.88', '', locale='vi_VN', format=u'.#0')
    # vn = format_currency(decimal.Decimal( "1232112322323.88" ), '',locale='vi_VN',)
    # print(n_str)
    # print(vn)
    # emails = re.findall(r'[\d\.-]+[.|,\d][\d\.-]+', abc)
    # for email in emails:
    #     print(email)
    # test_word_to_num(10)
    # num = word_to_num('Năm mươi bảy ngàn hai trăm sáu mươi Đô la Mỹ')  # 37.260 USD
    num = word_to_num(
        'Số tiền viết bằng chữ chín tỉ, bốn trăm năm mươi tám triệu, bảy trăm linh hai nghìn, hai trăm ba mươi hai đồng')  # 37.260 USD
    print(num)
    # get_money_fr_str(abc)
