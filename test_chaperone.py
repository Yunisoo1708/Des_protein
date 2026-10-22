from pathlib import Path

from chaperone import ExecutionEngine, init_cli

init_cli() #enables testing environment and --partition
root = Path(__file__).resolve().parent #Resolve convert the path into an absolute object
engine = ExecutionEngine(rundir=root/'test_run_chaperone')

backbones = engine.run('rfd3', 
                       inputs=str(root/'json_prep'/'5YXH.json'), 
                       batch_size = 3, 
                       n_batches = 2,
                       compile_model = False,) #Input requires string. compile off: container tmpfs is only 64MB

sequence = engine.run('foundry_mpnn',
                      data = backbones,
                      model_type = 'protein_mpnn',
                      num_seq_per_pdb = 1,)

predictions = engine.run('rf3', data=sequence, diffusion_batch_size=1)


#Unpack results. Unpack should be only used when you want a table for inspection or export
t_backbones =backbones.unpack()
df = predictions.unpack()
print(df.head())
(root/'real_run').mkdir(exist_ok=True)
df.write_parquet(root/'real_run'/'results_chaperone_test.parquet')


