import subprocess
from pathlib import Path

from xml.parsers.expat import ExpatError

from tqdm import tqdm

from ase.io import read, write
from ase import Atoms as ase_Atoms

from jarvis.core.atoms import Atoms as jarvis_Atoms
from jarvis.db.figshare import data
from jarvis.io.vasp.outputs import Vasprun

output_dir = Path("DFPT_VASP_outputs")
output_dir.mkdir(exist_ok=True)

def has_dfpt(ent):
    raw_files = ent["raw_files"] 
    for fn in raw_files:
        if "DFPT" in fn:
            return True
    return False

def get_ase_atoms(ent):
    jat = jarvis_Atoms.from_dict(ent["atoms"])
    at = jat.ase_converter()

    #props_to_save = ["spg_number", "spg_symbol", "formula", "formation_energy_per_atom", "func", "optb88vdw_total_energy",
    #                 "epsx", "epsy", "epsz", "dfpt_piezo_max_dielectric", "dfpt_piezo_max_dielectric_ionic",
    #                 "ehull", "reference", "magmom_oszicar", "mepsx", "mepsy", "mepsz", "crys", "modes", "max_ir_mode", "typ", "dimensionality"]

    #at.info["jid"] = ent["jid"]

    #if ent["jid"] == "JVASP-28474":
    #    import pdb; pdb.set_trace()


    #for prop in props_to_save:
    #    if prop in ent:
    #        at.info[f"jarvis_{prop}"] = ent[prop]
    
    #if ent["jid"]=="JVASP-1372":
        #import pdb; pdb.set_trace()
    #get_data(ent)

    for key, val in ent.items():
        at.info[f"jarvis_{key}"] = val

    return at

def get_data(ent, output_dir=output_dir):

    vasp_xml = download_vasp_outputs(ent["raw_files"], output_dir)
    try:
        vrun = Vasprun(vasp_xml)
    except ExpatError:
        print(f"------------------------------\n{vasp_xml}-------------------------------\n")



def download_vasp_outputs(raw_files, output_dir):

    for fn in raw_files:
        if "DFPT" in fn:
            fn = fn.split(",")
            file_link = fn[2]
            out_fn = fn[1]
            #print(file_link, out_fn)
            break


    vasp_output = output_dir / out_fn.replace(".zip", ".vasprun.xml")
    outcar = output_dir / out_fn.replace(".zip", ".OUTCAR")

    if vasp_output.exists() and outcar.exists():
        print(f"skipping {out_fn}")
        return vasp_output, outcar

    download_out_fn = output_dir / out_fn
    command = f"wget {file_link} -O {str(download_out_fn)}"
    subprocess.run(command, shell=True)
    
    if not vasp_output.exists():
        unzip_command = f'unzip -j {str(download_out_fn)} "vasprun.xml" -d {output_dir} && mv {output_dir/"vasprun.xml"} {vasp_output}'
        subprocess.run(unzip_command, shell=True)

    if not outcar.exists():
        unzip_command = f'unzip -j {str(download_out_fn)} "OUTCAR" -d {output_dir} && mv {output_dir/"OUTCAR"} {outcar}'
        subprocess.run(unzip_command, shell=True)


    # remove file
    download_out_fn.unlink() 
    
    return vasp_output, outcar


dft_3d = data(dataset="dft_3d")

ase_atoms = []
for ent in tqdm(dft_3d):
    if has_dfpt(ent):
        ase_atoms.append(get_ase_atoms(ent))
#ase_atoms = [get_ase_atoms(ent) for ent in dft_3d if has_dfpt(ent)]

write("jarvis_dfpt_data.xyz", ase_atoms)

print("done")


