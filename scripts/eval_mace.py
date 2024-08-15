import numpy as np
from pathlib import Path 

from tqdm import tqdm

from ase.io import read, write

from wfl.calculators import generic
from wfl.configset import ConfigSet, OutputSpec
from wfl.autoparallelize import AutoparaInfo

from mace.calculators.mace import MACECalculator 

mace_path = "/raven/u/egg/work/13.efg_fitting/2.full_data_initial_fits/models/EFGsMACE_test.model"

test_fn_in = "/raven/u/egg/data/lto_efgs_from_ange/LTO42-ratttle-pristine-fracscale_validate_1312_b73de8.xyz"
train_fn_in = "/raven/u/egg/data/lto_efgs_from_ange/LTO42-ratttle-pristine-fracscale_train_1312_b73de8.xyz"

test_fn_out = "test.mace.xyz"
train_fn_out = "train.mace.xyz"

#ci = ConfigSet([test_fn_in, train_fn_in])
#os = OutputSpec([test_fn_out, train_fn_out])
# 
# ci = ConfigSet(test_fn_in)
# os = OutputSpec(test_fn_out)
# 
# 
calc_params = {
    "device": "cpu",
    "model_type" : "EFGsMACE",
    }
#         
# calc =  (MACECalculator, [mace_path], calc_params)
# 
# autopara_info = AutoparaInfo(num_python_subprocesses=1)
# 
# generic.calculate(ci, os, calc, properties=["efgs"], output_prefix="mace_", autopara_info=autopara_info)
# 

calc = MACECalculator(mace_path, **calc_params)

for fin, fout in zip([train_fn_in, test_fn_in], [train_fn_out, test_fn_out]):

    if Path(fout).exists():
        print(f"file {fout} exists, skipping")
        continue


    ats_in = read(fin, ":")
    ats_out = []
     
    for at in tqdm(ats_in):

        at.calc = calc
        at.calc.calculate(atoms=at)
        efgs = at.calc.results["efgs"]
        at.arrays["mace_efg"] = efgs.reshape((-1, 9))
        ats_out.append(at)

    write(fout, ats_out)




