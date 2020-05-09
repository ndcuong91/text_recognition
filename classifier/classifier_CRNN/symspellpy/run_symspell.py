import pkg_resources
from symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "freq_name_dic.txt")
bigram_path = pkg_resources.resource_filename(
    "symspellpy", "freq_name_bigram.txt")
# term_index is the column of the term and count_index is the
# column of the term frequency
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1, encoding='utf-8')
# sym_spell.load_dictionary('C:/Users/nt.anh6/PycharmProjects/aicr_vn/nlp_model/spell_checker/dict/vi_full.txt', term_index=0, count_index=1, encoding='utf-8')
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2, encoding='utf-8')

# lookup suggestions for multi-word input strings (supports compound
# splitting & merging)
input_term = "Ngyễn tành nm"
# max edit distance per lookup (per single word, not per whole input string)
# suggestions = sym_spell.lookup(input_term, Verbosity.ALL, max_edit_distance=2, include_unknown=True)
suggestions = sym_spell.lookup_compound(input_term, max_edit_distance=2)
# display suggestion term, edit distance, and term frequency
for suggestion in suggestions:
    print(suggestion)

def load_name_corection(dictionary_path, bigram_path):
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    # dictionary_path = pkg_resources.resource_filename(
    #     dictionary_path)
    # bigram_path = pkg_resources.resource_filename(
    #     bigram_path)
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1, encoding='utf-8')
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2, encoding='utf-8')
    return sym_spell

def correct_name(sym_spell, name_string=None):
    suggestions = sym_spell.lookup_compound(input_term, max_edit_distance=2)
    # display suggestion term, edit distance, and term frequency
    ouput = []
    for suggestion in suggestions:
        # print(suggestion)
        ouput.append(suggestion._term)
    return ouput

if __name__ == "__main__":
    sym_spell = load_name_corection("freq_name_dic.txt", "freq_name_bigram.txt")
    print(correct_name(sym_spell, input_term))
