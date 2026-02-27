#PBS -l nodes=inspur13:ppn=64
#PBS -q workq
#PBS -N outer_40_p

cd   $PBS_O_WORKDIR
echo $PBS_NODEFILE

ulimit -s unlimited

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:\
/sharehome/test/.local/comsol56/multiphysics/lib/glnxa64:\
/sharehome/test/.local/comsol56/multiphysics/ext/graphicsmagick/glnxa64:\
/sharehome/test/.local/comsol56/multiphysics/ext/cadimport/glnxa64:\
/sharehome/test/.local/comsol56/multiphysics/lib/glnxa64/gcc:

cd $OLDPWD/generate_data
python3 cross_shape_kpa.py | tee temp/log_train
# python3 double_layer_kpa.py | tee temp/log_train





