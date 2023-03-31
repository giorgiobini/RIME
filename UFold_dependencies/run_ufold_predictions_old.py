import os 
import time
import subprocess
from pathlib import Path

def main():
    chunks = [file for file in os.listdir(files_dir) if file.startswith('chunk')]
    input_path =  os.path.join(files_dir, 'input.txt')
    result_path = os.path.join(results_dir, 'input_dot_ct_file.txt')
    dirs_to_remove = [os.path.join(results_dir,'save_ct_file'), os.path.join(results_dir,'save_varna_fig')]
    for chunk in chunks:
        start_time_chunk = time.time()


        chunk_path = os.path.join(files_dir, chunk)


        cmd0 = ['mv', chunk_path, input_path] #rename
        cmd1 = ['python', os.path.join(ROOT_DIR, 'UFold_dependencies', 'ufold_predict.py')]
        cmd2 = ['rm', input_path] #remove
        cmd3 = ['mv', result_path, os.path.join(results_dir, chunk)]
        cmd4 = ['rm', '-r', dirs_to_remove[0]]
        cmd5 = ['rm', '-r', dirs_to_remove[1]]
        commands = [cmd0, cmd1, cmd2, cmd3, cmd4, cmd5]


        for command in commands:
            subprocess.call(command, shell=False)


        total_minutes_chunk = (time.time() - start_time_chunk)/60
        print('{} done in {} minutes'.format(chunk, total_minutes_chunk))

if __name__ == '__main__':
    #run me with: -> nohup python run_ufold_predictions.py &> run_ufold_predictions.out &
    ROOT_DIR = os.path.dirname(os.path.abspath('.'))
    files_dir = os.path.join(ROOT_DIR, 'UFold_dependencies', 'data')
    results_dir = os.path.join(ROOT_DIR, 'UFold_dependencies', 'results')
    start_time = time.time()
    main()
    total_seconds = time.time() - start_time
    total_minutes = total_seconds/60
    print('Done in {} minutes'.format(total_minutes))