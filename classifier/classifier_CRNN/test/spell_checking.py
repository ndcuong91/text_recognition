from symspellpy.address_spell_check import correct_address, load_address_correction
import pickle, os


db_file = 'db.pickle'
# if os.path.exists(db_file) is False:
#     load_address_correction('database.pickle', db_file)

with open(db_file, 'rb') as handle:
    db = pickle.load(handle)
    # inp = 'Số 83 Thanh Lân, Phường hun T, Quận Hoang Mi, Hàn Nội'
inp = 'Kim Mần, Xã Xín Mồn, Huyện Xín Mần, Hà Giang'

# print(input_)
def make_input(inp):
    inp = inp.split(',')
    inp = [i.rstrip().lstrip() for i in inp]
    output = {}
    output['city'] = None
    output['district'] = None
    output['ward'] = None
    output['street'] = None
    if len(inp) > 3:
        output['city'] = inp[-1]
        output['district'] = inp[-2]
        output['ward'] = inp[-3]
        output['street'] = inp[-4]
    elif len(inp) == 3:
        output['city'] = inp[-1]
        output['district'] = inp[-2]
        output['ward'] = inp[-3]
    elif len(inp) == 2:
        output['city'] = inp[-1]
        output['district'] = inp[-2]
    elif len(inp) == 1:
        output['city'] = inp[-1]
    return output

csv = 'dvhcvn.csv'
inp = make_input(inp)
out = correct_address(db=db, csv_file=csv,
    city=inp['city'], 
    district=inp['district'], 
    ward=inp['ward'], 
    street='Quang Trung'
    )
print(out)