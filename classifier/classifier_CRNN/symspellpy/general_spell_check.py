import collections
import datetime
import pandas as pd
import re
from fuzzywuzzy import fuzz

try:
    from . import SymSpell, Verbosity
    from .unicode_utils import *
except:
    from symspellpy import SymSpell, Verbosity
    from unicode_utils import *

dif_day = [['1', '4'], ['1', '7'], ['3', '5']]


def export_freq_dic(words_prov):
    c = collections.Counter()
    words = list(words_prov)
    for i in words:
        c.update(set(i))
    unigram_list = []
    for i in c:
        if i != '' and i != ' ':
            unigram_list.append([i, str(c[i])])
    return unigram_list


def export_freq_bigram(list_prov):
    c = collections.Counter()
    for i in list_prov:
        c.update(set(zip(i[:-1], i[1:])))
    bigram_list = []
    for i in c:
        list_i = list(i)
        list_i.append(c[i])
        bigram_list.append(list_i)
    return bigram_list


def correct_capital(raw, fixed):
    if fixed is not None:
        fixed = fixed.replace('_', ' ')
        fixed_li = fixed.split(' ')
        if len(fixed_li) > 1:
            unknow = []
            for i, w in enumerate(fixed_li):
                group = re.search(w, raw, re.IGNORECASE)
                if group is not None:
                    if group.group().islower():
                        fixed_li[i] = w.lower()
                    elif group.group().isupper():
                        fixed_li[i] = w.upper()
                    else:
                        fixed_li[i] = w.capitalize()
                else:
                    unknow.append(i)
            for i in unknow:
                if i >= 1:
                    if fixed_li[i - 1].isupper():
                        fixed_li[i] = fixed_li[i].upper()
                    elif fixed_li[i - 1].islower():
                        fixed_li[i] = fixed_li[i].lower()
                    else:
                        fixed_li[i] = fixed_li[i].capitalize()
                else:
                    if fixed_li[i + 1].isupper():
                        fixed_li[i] = fixed_li[i].upper()
                    elif fixed_li[i + 1].islower():
                        fixed_li[i] = fixed_li[i].lower()
                    else:
                        fixed_li[i] = fixed_li[i].capitalize()
            return ' '.join(fixed_li)
        else:
            if raw.islower():
                fixed = fixed.lower()
            elif raw.isupper():
                fixed = fixed.upper()
            else:
                fixed = fixed.capitalize()
            return fixed
    else:
        return fixed


def load_name_corection(dictionary_path, bigram_path):
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1, encoding='utf-8')
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2, encoding='utf-8')
    return sym_spell


def correct_name(sym_spell, name_string=None):
    name_string = name_string.replace(' ', '_')
    suggestions = sym_spell.lookup_compound(name_string, max_edit_distance=2)
    # display suggestion term, edit distance, and term frequency
    list_name = []
    for suggestion in suggestions:
        list_name.append(suggestion._term)
    output = []
    for name in list_name:
        if name is not None:
            name = correct_capital(raw=name_string, fixed=name)
        else:
            name = name_string
        output.append(name)
    return output


def load_country_correction(csv_data, edit_distance=5, prefix_length=7):
    with open(csv_data, 'rb') as f:
        data = f.readlines()
    data = [ctr.decode("utf-8").replace('\r\n', '').lower().replace(' ', '_') for ctr in data]
    c = collections.Counter()
    for i in data:
        c.update({i})
    unigram = []
    for i in c:
        if i == 'việt_nam':
            c[i] = 1000
        unigram.append([i, str(c[i])])
    sym_spell = SymSpell(max_dictionary_edit_distance=edit_distance, prefix_length=prefix_length)
    sym_spell.load_dictionary_from_list(unigram, term_index=0, count_index=1)
    return sym_spell


def correct_country(raw, sym_spell, edit_distance=5, verbosity=0):
    countr_fix_lookup = sym_spell.lookup_compound(raw, max_edit_distance=edit_distance)
    # for ctr in countr_fix_lookup:
    #     print('lookup',ctr)
    if len(countr_fix_lookup) > 0:
        output = correct_capital(raw=raw, fixed=countr_fix_lookup[0]._term)
    else:
        output = raw
    # countr_fix_lookup_compound = sym_spell.lookup_compound(raw,max_edit_distance=edit_distance)
    # for ctr in countr_fix_lookup_compound:
    #     print('compou', ctr)
    return output


def load_date_correction(csv_data=None, edit_distance=2, prefix_length=4):
    data = csv_data
    data = [str(i) for i in data]
    data.append('  -  ')
    c = collections.Counter()
    for i in data:
        c.update({i})
    unigram = []
    for i in c:
        unigram.append([i, str(c[i])])
    sym_spell = SymSpell(max_dictionary_edit_distance=edit_distance, prefix_length=prefix_length)
    sym_spell.load_dictionary_from_list(unigram, term_index=0, count_index=1)
    return sym_spell


def correction(raw, sym_spell, edit_distance=2, verbosity=0):
    check_raw = raw.replace(' ', '')
    if check_raw != '':
        countr_fix_lookup = sym_spell.lookup(raw.lower(), max_edit_distance=edit_distance, verbosity=verbosity)
        if len(countr_fix_lookup) > 1:
            output = countr_fix_lookup[0]._term
            for name in countr_fix_lookup:
                # print(name._term)
                if '-' in raw:
                    day_raw, month_raw = raw.split('-')
                    day_fix, month_fix = name._term.split('-')
                    difday = findUncommonChars(day_fix, day_raw)
                    for d in dif_day:
                        if d[0] in difday and d[1] in difday:
                            output = name._term


        elif len(countr_fix_lookup) == 1:
            output = countr_fix_lookup[0]._term
        else:
            output = raw
        return output
    else:
        return ''


def correct_date(day, month, year):
    fixed_year = year
    if year.strip() != '':
        year_corpus = datetime.datetime.today().year
        year_corpus = list(range(year_corpus, year_corpus - 100, -1))
        year_spell = load_date_correction(csv_data=year_corpus)
        fixed_year = correction(year, year_spell)

    fixed_day = day
    fixed_month = month
    if day.strip() != '' and month.strip() != '':
        datelist = pd.date_range(start='2020-01-01', end='2020-12-31').tolist()
        datelist = list(map(pd.Timestamp.to_pydatetime, datelist))
        datelist = [d.strftime("%d-%m") for d in datelist]
        date_spell = load_date_correction(csv_data=datelist, edit_distance=5, prefix_length=6)
        fixed_date = correction(day + '-' + month, date_spell, edit_distance=5)
        fixed_day, fixed_month = fixed_date.split('-')

    return fixed_day, fixed_month, fixed_year


def load_cpn_corection(companies_list, debug=False):
    with open(companies_list, 'r', encoding='utf-8') as f:
        l = f.read()
    l = l.lower()
    l = l.split('\n')
    m = []
    for w in l:
        m.append(w.split())
    bi = export_freq_bigram(m)
    uni = export_freq_dic(m)
    if debug:
        print(uni)
        print(bi)
    sym_spell = SymSpell(max_dictionary_edit_distance=5, prefix_length=7)
    sym_spell.load_dictionary_from_list(uni, term_index=0, count_index=1)
    sym_spell.load_bigram_dictionary_from_list(bi, term_index=0, count_index=2)
    return sym_spell


def correct_cpn(name_string, sym_spell, debug=False, GT_file=None):
    suggestions = sym_spell.lookup_compound(name_string.lower(), max_edit_distance=4)
    # display suggestion term, edit distance, and term frequency
    list_name = []
    for suggestion in suggestions:
        list_name.append(suggestion._term)
    if debug: print('list_name', list_name)

    if GT_file:
        with open(GT_file, 'r', encoding='utf-8') as f:
            l = f.read()
        l = l.lower()
        l = l.split('\n')
        distant = 0
        relation_fixed = suggestions[0]._term
        for ctr in l:
            tem_distant = fuzz.ratio(suggestions[0]._term, ctr)
            if tem_distant > distant:
                distant = tem_distant
                relation_fixed = ctr
        relation_fixed = correct_capital(raw=name_string, fixed=relation_fixed)
        return relation_fixed, fuzz.ratio(name_string, relation_fixed)
    output = []
    for name in list_name:
        if name is not None:
            name = correct_capital(raw=name_string, fixed=name)
        else:
            name = name_string
        output.append(name)
    return output[0], fuzz.ratio(name_string, output[0])


smallCurrencies = [
    "cents",
    "xu"
]
bigCurrencies = [
    "đô la mỹ",
    "việt nam đồng",
    "vnd",
    "vnđ",
    "usd",
    "krw",
    "đồng",
    "mỹ",
    "đô",
    "la"
]


def correctMoneyByWords(money_string, sym_spell, debug=False, GT_file=None):
    existSmallCurrency = False
    money_string_fixed = correct_cpn(money_string, sym_spell, debug=debug, GT_file=GT_file)
    # print(name_string)
    money_string_fixed, conf = money_string_fixed
    for sc in smallCurrencies:
        group = re.search(sc, money_string_fixed, re.IGNORECASE)
        if group is not None:
            money_string_fixed = money_string_fixed.split(group.group())[0].strip() + ' ' + group.group()
            existSmallCurrency = True
    if existSmallCurrency == False:
        for p in bigCurrencies:
            group = re.search(p, money_string_fixed, re.IGNORECASE)
            if group is not None:
                money_string_fixed = money_string_fixed.split(group.group())[0].strip() + ' ' + group.group()
    temp_dt = fuzz.ratio(decompound_unicode(money_string_fixed), decompound_unicode(money_string[0]))
    # print('temp_dt:',temp_dt)
    # print('money_string[0]:',money_string[0], 'money_string_fixed', money_string_fixed)
    if temp_dt < 30:
        return (money_string, conf)
    else:
        return (money_string_fixed, conf)


def load_relationship_correction(csv_data, edit_distance=5, prefix_length=7):
    with open(csv_data, 'rb') as f:
        data = f.readlines()
    data = [ctr.decode("utf-8").replace('\r\n', '').lower().replace(' ', '_') for ctr in data]
    data = [decompound_unicode(ctr) for ctr in data]
    uni = export_freq_dic([data])
    sym_spell = SymSpell(max_dictionary_edit_distance=edit_distance, prefix_length=prefix_length)
    sym_spell.load_dictionary_from_list(uni, term_index=0, count_index=1)
    return sym_spell


def correct_relationship(raw, sym_spell, edit_distance=5, verbosity=0, debug=False):
    check_raw = raw.replace(' ', '')
    if check_raw != '':
        raw = raw.lstrip().rstrip().replace(' ', '_').lower()
        if debug: print(raw)
        relation_fix_lookup = sym_spell.lookup(raw, max_edit_distance=edit_distance, verbosity=0)
        distant = 0
        relation_fixed = None
        for ctr in relation_fix_lookup:
            if fuzz.ratio(raw, ctr._term) > distant:
                distant = fuzz.ratio(raw, ctr._term)
                relation_fixed = ctr._term
            if debug:
                print('lookup', ctr, 'fuzzy distance', fuzz.ratio(raw, ctr._term))
        if relation_fixed:
            output = correct_capital(raw=raw, fixed=relation_fixed)
        else:
            output = raw
        return output
    else:
        return ''


def findUncommonChars(str1, str2, MAX_CHAR=26):
    present = [0] * MAX_CHAR
    for i in range(0, MAX_CHAR):
        present[i] = 0
    l1 = len(str1)
    l2 = len(str2)
    for i in range(0, l1):
        present[ord(str1[i]) - ord('0')] = 1
    for i in range(0, l2):
        if (present[ord(str2[i]) - ord('0')] == 1 or
                present[ord(str2[i]) - ord('0')] == -1):
            present[ord(str2[i]) - ord('0')] = -1
        else:
            present[ord(str2[i]) - ord('0')] = 2
    ret = []
    for i in range(0, MAX_CHAR):
        if present[i] == 1 or present[i] == 2:
            # print(chr(i + ord('0')), end=" ")
            ret.append(chr(i + ord('0')))
    return ret


if __name__ == "__main__":
    cpn_list = 'data/companies-list.txt'
    cpn_spell = load_cpn_corection(cpn_list, True)
    print(correct_cpn('CÔNG TY TNHH SXDVCÔNG  VIỆT NAM', cpn_spell, GT_file=cpn_list, debug=True))
    # print('fuzzy distance', fuzz.ratio('CÔNG TY TNHH SXDVCÔNG NGHỆ BÁN DÃN TOÀN CẨU VIỆT NAM',
    #                                    'CÔNG TY TNHH SX DV CÔNG NGHỆ BÁN DẪN TOÀN CẦU VIỆT NAM'))
    print(correct_date('1', '', '12'))
    #
    # cntry_list = 'data/country-list.txt'
    # sym_spell = load_country_correction(csv_data=cntry_list)
    # print(correct_country('vnv', sym_spell))
    #
    # csv_data = 'data/relationship_list.txt'
    # sym_spell = load_relationship_correction(csv_data=csv_data)
    # print(correct_relationship('Bn Gái', sym_spell, debug=True))
    # input_term = "Nguyễn, Minh Thanh"
    # sym_spell = load_name_corection("freq_name_dic.txt", "freq_name_bigram.txt")
    # print(correct_name(sym_spell, input_term))
    #
    # money_word_list = 'data/num_by_word.txt'
    # money_word_spell = load_cpn_corection(money_word_list, debug=True)
    # print(correct_cpn('Bảy mươi tám ngìn chín trăm chín mươi tám Đô la Mi và năm mươi m0E CCNES', money_word_spell))
    # print(correct_capital(raw='Anw', fixed='anw'))
    #
    # str1 = "74-03"
    # str2 = "24-03"
    # findUncommonChars(str1, str2)
    # print(findUncommonChars(str1, str2))
