import os

def CURRENT_DIR():
    cwd = os.getcwd()
    return os.chdir(cwd[:(cwd.index("Eliza")+5)])

CURRENT_DIR()

os.getcwd()

# from preprocessing import input_preprocess

# data = {
#     "plot_area": 200,
#     "habitable_surface":196,
#     "land_surface":123,
#     "room_count":5,
#     "bedroom_count":3
# }

# new_content = input_preprocess.input_data(source=data)

# print(new_content)


