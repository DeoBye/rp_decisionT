import json
from pathlib import Path

# {
#     "projection_size": 10,   # 8 - 16
#     "projection_count": 200,   # 4000
#     "ensemble_size": 16,   # 10 - 20
#     "classifier": "dt"
# }

if __name__ == "__main__":
    for proj_size in range(8, 17):
        for ensem_size in range(10, 21):
            file_name = 'proj_size_' + str(proj_size) + '_ensem_size_' + str(ensem_size) 
            file_path = Path('./config/10000') / Path(file_name).with_suffix('.cfg').name
            config = {"projection_size": int(proj_size),
                      "projection_count": int(4000),
                      "ensemble_size": int(ensem_size),
                      "max_it": 10000,
                      "classifier": "dt"}
            with open(file_path, 'w') as fp:
                json.dump(config, fp)
                
            fp.close()