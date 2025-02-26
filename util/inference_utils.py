from Bio.SeqIO.FastaIO import FastaIterator
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

def ParseFasta(filename):
    
    all_ids=[]
    all_seq=[]
    
    with open(filename) as handle:
        for record in FastaIterator(handle):
            #print(str(entry.id)) #This is header of fasta entry
            #print(str(entry.seq))
            # all_ids+=[str(record.id)]
            # all_seq+=[str(record.seq)]
            all_ids.append(str(record.id))
            all_seq.append(str(record.seq).replace('U', 'T'))  # Convert U to T


    res_fasta = pd.DataFrame([all_ids,all_seq]).T
    res_fasta.columns =["id","seq"]
    res_fasta.loc[:,"length"]=res_fasta.loc[:,"seq"].apply(lambda x: len(x))
    res_fasta.loc[:,"header"]=res_fasta.loc[:,"id"]
    
    res_fasta=res_fasta[["header","seq","id","length"]]
    return res_fasta

def Dirs(dir_out):
    if os.path.exists(dir_out) == False:
            os.makedirs(dir_out)

class RIME_inference():
    
    def __init__(self,bin_bedtools,query_fasta,target_fasta,dir_out,name_analysis):
        
        self.query_fasta = query_fasta
        self.target_fasta = target_fasta
        self.name_analysis = name_analysis
        
        self.dir_out = dir_out+name_analysis+"/"
        self.bin_bedtools = bin_bedtools
        
        self.emb = None
        self.emb_bedfile = None
        self.wind = None
        self.wind_bedfile = None
        self.pairs = None
        
        Dirs(self.dir_out)
           
    def Windows(self,size_window=200,step=1,mode="generate"):
        
        sub_dir_out=self.dir_out+"windows/"+str(size_window)+"_"+str(step)+"/"
        Dirs(sub_dir_out)
        
        genome_file=sub_dir_out+"genome.txt"
        file_window_temp=sub_dir_out+"window_temp.txt"
        
        if mode == "generate":
            
            mode_inputs = ["q","t"]
            for mode_input in mode_inputs:
                
                print("processing "+mode_input)
            
                if mode_input == "q":
                    res_fasta = ParseFasta(self.query_fasta)
                    fasta_file=self.query_fasta
                if mode_input == "t":
                    res_fasta = ParseFasta(self.target_fasta)
                    fasta_file=self.target_fasta
    
                res_fasta.loc[:,["id","length"]].to_csv(genome_file,sep="\t",index=None,header=None)
                os.system(self.bin_bedtools+" makewindows -i srcwinnum  -g "+genome_file+" -w "+str(size_window)+" -s "+str(step)+" > "+file_window_temp)
                print(self.bin_bedtools+" makewindows -i srcwinnum  -g "+genome_file+" -w "+str(size_window)+" -s "+str(step)+" > "+file_window_temp)

                wind_df=pd.read_table(file_window_temp,header=None)
                wind_df.loc[:,"score"]=1
                wind_df.loc[:,"strand"]="+"

                wind_df.columns=["chrom","start","end","name","score","strand"]

                # filtro minimo
                wind_df.loc[:,"width"]=wind_df.loc[:,"end"]-wind_df.loc[:,"start"]
                wind_df=wind_df.loc[wind_df["width"]>30]

                # se window included test
                wind_df=wind_df.loc[wind_df.groupby(["chrom","end"])["width"].idxmax()]

                del wind_df["width"]
                
                wind_df.loc[:,"group"]=mode_input

                if mode_input == "q":
                    wind_df_final = wind_df.copy(deep=True)
                if mode_input == "t":
                    wind_df_final = wind_df_final.append(wind_df)
                
            print(file_window_temp)
            wind_df_final.loc[:,"name"]=wind_df_final.loc[:,"name"]+"__"+wind_df_final.loc[:,"start"].astype(str)+"_"+wind_df_final.loc[:,"end"].astype(str)
        
            wind_df_final.to_csv(file_window_temp,sep="\t",header=None,index=None)
            
        if mode == "load":
            
            wind_df_final=pd.read_table(file_window_temp,sep="\t",header=None,index_col=0)
            print(wind_df_final)
            wind_df_final.columns=["chrom","start","end","name","score","strand"]
        
        self.wind = wind_df_final
        self.wind_bedfile = file_window_temp
        
        return wind_df_final
    
    
    def Embedding(self,length_max_embedding=5970,step="standard",mode="generate"):
        
        print("\t")
        print("mode: generate or load")
        
        sub_dir_out=self.dir_out+"embeddings/"+str(length_max_embedding)+"/"
        Dirs(sub_dir_out)
        
        # lo step è l'embedding diviso 2
        
        if step == "standard":
            step = int(length_max_embedding/2)
            print("step defined as half the embedding length")
        
        print("length emb:",length_max_embedding)
        print("step for emedding:",step)

        file_embedding_out=self.dir_out+"embedding_query.csv"

        if mode == "generate":
            
            mode_inputs = ["q","t"]
            for mode_input in mode_inputs:
                
                print("processing "+mode_input)
            
                if mode_input == "q":
                    res_fasta = ParseFasta(self.query_fasta)
                    fasta_file=self.query_fasta
                if mode_input == "t":
                    res_fasta = ParseFasta(self.target_fasta)
                    fasta_file=self.target_fasta
            
                file_embedding_temp = sub_dir_out+"embedding_temp.bed"
                
                genome_file=sub_dir_out+"genome.txt"

                res_fasta.loc[:,["id","length"]].to_csv(genome_file,sep="\t",index=None,header=None)

                os.system(self.bin_bedtools+" makewindows -i srcwinnum  -g "+genome_file+" -w "+str(length_max_embedding)+" -s "+str(step)+" > "+file_embedding_temp)

                emb_df=pd.read_table(file_embedding_temp,header=None)
                emb_df.loc[:,"width"]=emb_df.loc[:,2]-emb_df.loc[:,1]
                
                #print(emb_df)
                
                # filtro minimo
                emb_df=emb_df.loc[emb_df["width"]>30]
                
                # se embedding esce coi limiti allora lo ricalibro prendendo le ultime 5970 basi
                emb_df.loc[(emb_df[3].str.count("_1")==0)&(emb_df["width"]<length_max_embedding),1]=emb_df.loc[(emb_df[3].str.count("_1")==0)&(emb_df["width"]<length_max_embedding),2]-length_max_embedding
                emb_df.loc[emb_df[1]<0,1]=0
                
                # se due embedding in uno stesso trascritto "0" condividono le coordinate finali allora seleziono il più grande
                emb_df=emb_df.loc[emb_df.groupby([0,2])["width"].idxmax()]
                
                print(emb_df)
                
                del emb_df["width"]
                emb_df.loc[:,"score"]=1
                emb_df.loc[:,"strand"]="+"
                
                emb_df.to_csv(file_embedding_temp,sep="\t",index=None,header=None)
                
                print(file_embedding_temp)
                
                
                ### generate fasta
                os.system(self.bin_bedtools+" getfasta -name -s -fi "+fasta_file+" -bed "+file_embedding_temp+" > "+file_embedding_temp.replace(".bed",".fa"))
                emb_fasta = ParseFasta(file_embedding_temp.replace(".bed",".fa"))
                
                #print(emb_fasta)
                emb_fasta.loc[:,"group"]=mode_input
                
                emb_df.loc[:,"group"]=mode_input
                
                if mode_input == "q":
                    
                    emb_fasta_final = emb_fasta.copy(deep=True)
                    emb_bed_final = emb_df.copy(deep=True)
                    
                if mode_input == "t":
                    
                    emb_bed_final = emb_bed_final.append(emb_df) 
                    emb_fasta_final = emb_fasta_final.append(emb_fasta)
                    
            emb_bed_final.to_csv(sub_dir_out+"temp_embedding.bed",sep="\t",index=None,header=None)
        
            emb_fasta_final.loc[:,"header"]=emb_fasta_final.loc[:,"header"].apply(lambda x: x.replace(">","").split("::")[0])
            emb_fasta_final.columns=["id_query","cdna","info_name","length_seq","group"]
            emb_fasta_final.drop_duplicates().reset_index().to_csv(file_embedding_out,sep=",",index=None)
            
            #os.system("rm -r "+sub_dir_out)
                
        if mode == "load": 
            
            emb_fasta_final=pd.read_table(file_embedding_out,sep=",",index_col=0)

        #print("N seq:",len(set(emb_fasta_final.loc[:,"id_query"])))
        #print("N emedding:",len(set(emb_fasta_final.loc[:,"name"])))
    
        self.emb = emb_fasta_final
        self.emb_bedfile =sub_dir_out+"temp_embedding.bed"
            
        return emb_fasta_final
    
    def AssembleQueryTargetRanges(self):
        
        #bed_wind="/data01/users_space/adriano/projects/my_projects/rnarna_paper_files/test_rinet/window_based/NORAD/rinet_preprocess/windows/200_100/window_temp.txt"
        #bed_emb="/data01/users_space/adriano/projects/my_projects/rnarna_paper_files/test_rinet/window_based/NORAD/rinet_preprocess/embeddings/embedding_temp.bed"
    
        bed_wind=self.wind_bedfile
        bed_emb=self.emb_bedfile
        
        #file_embedding_out.replace(".csv",".bed")
        os.system(self.bin_bedtools+" intersect -f 1 -s -wao -a "+bed_wind+" -b "+bed_emb+" > "+self.dir_out+"temp.bed")

        wind_emb=pd.read_table(self.dir_out+"temp.bed",header=None)
        wind_emb.columns=["chrom_wind","start_wind","end_wind","name_wind","score_wind","strand_wind","type_wind","chrom_emb","start_emb","end_emb","name_emb","score_emb","strand_emb","type_emb","coverage"]
        wind_emb.loc[:,"chrom"]=wind_emb.loc[:,"name_emb"]
        
        wind_emb.loc[:,"start"]=wind_emb.loc[:,"start_wind"]-wind_emb.loc[:,"start_emb"]
        wind_emb.loc[:,"end"]=wind_emb.loc[:,"end_wind"]-wind_emb.loc[:,"start_emb"]
        wind_emb.loc[:,"name"]=wind_emb.loc[:,"name_wind"]
        wind_emb.loc[:,"score"]=1
        wind_emb.loc[:,"strand"]=wind_emb.loc[:,"strand_wind"]
        
        
        wind_emb=wind_emb[["chrom","start","end","name","score","strand","type_wind"]]

        w_q=wind_emb.loc[wind_emb["type_wind"]=="q",["chrom","start","end","name"]]
        w_t=wind_emb.loc[wind_emb["type_wind"]=="t",["chrom","start","end","name"]]

        w_q['key'] = 0
        w_t['key'] = 0
        w_all = pd.merge(w_q,w_t,left_on="key",right_on="key", how='outer',suffixes=["_1","_2"])

        w_all=w_all[["chrom_1","chrom_2","start_1","end_1","start_2","end_2","name_1","name_2"]]
        w_all=w_all.reset_index()
        w_all.columns=["id_pair","embedding1name","embedding2name","start_window1","end_window1","start_window2","end_window2","window_1","window_2"]

        w_all.to_csv(self.dir_out+"pairs.csv")

        self.pairs=w_all.copy(deep=True)
        
        os.system("rm "+self.dir_out+"temp.bed")
        
        return w_all

    # def LoadEmbedding(self,batch_size=1):
        
    #     # FILES
    #     dir_pred=self.dir_out
    #     pairs_file=dir_pred+"pairs.csv"
    #     embedding_file=dir_pred+"embedding_query.csv"

    #     ### EMBEDDINGS
    #     embedding_dir=dir_pred+"embeddings"
    #     file_out_embedding=dir_pred+"download_embeddings.out"
    #     conda_env = "rnarna"
    #     script_download_embedding="download_embeddings.py"
    #     program_to_run = "python "+script_download_embedding+" --batch_size="+str(batch_size)+" --path_to_embedding_query_dir="+dir_pred+" --embedding_dir="+embedding_dir+" &> "+file_out_embedding
    #     # Comando per attivare l'environment Conda e lanciare il programma
    #     command = f"conda run -n {conda_env} {program_to_run}"
    #     # Esegui il comando
    #     subprocess.run(command, shell=True, check=True)
        
    # def InferProbability(self,model_name="arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0086"):
        
    #     dir_pred=self.dir_out
    #     pairs_file=dir_pred+"pairs.csv"
    #     embedding_file=dir_pred+"embedding_query.csv"
        
    #     # Nome dell'environment Conda e comando da eseguire
    #     file_out_inference=dir_pred+"run_inference_new.out"
    #     conda_env = "dnabert"
    #     script_run_inference="run_inference_new.py"
    #     program_to_run = "python "+script_run_inference+" --pairs_path="+pairs_file+" --model_name="+model_name+" &> "+file_out_inference
    #     # Comando per attivare l'environment Conda e lanciare il programma
    #     command = f"conda run -n {conda_env} {program_to_run}"
    #     # Esegui il comando
    #     subprocess.run(command, shell=True, check=True)



def associateRIMEpobability(directory):
    
    file_probability_indexed=os.path.join(directory, "predictions.csv")
    
    prob=pd.read_table(file_probability_indexed, sep=",")

    pairs=pd.read_table(os.path.join(directory, "pairs.csv"),sep=",",index_col=0)

    pairs_prob=pd.merge(pairs,prob,left_on="id_pair",right_on="id_sample",how="left")
    pairs_prob=pairs_prob.fillna(0)

    ### 
    pairs_prob_mean=pd.DataFrame(pairs_prob.groupby(["window_1","window_2"])["probability"].mean())
    pairs_prob_mean=pairs_prob_mean.reset_index()

    pairs_prob.loc[:,"id_pair_window"]=pairs_prob.loc[:,"window_1"]+"_"+pairs_prob.loc[:,"window_2"]

    pairs_prob_mean.columns=["window_1","window_2","probability_mean"]
    pairs_prob_mean.loc[:,"id_window"]=pairs_prob_mean.loc[:,"window_1"]+"__"+pairs_prob_mean.loc[:,"window_2"]
    del pairs_prob_mean["window_1"]
    del pairs_prob_mean["window_2"]

    pairs_prob.loc[:,"id_window"]=pairs_prob.loc[:,"window_1"]+"__"+pairs_prob.loc[:,"window_2"]

    pairs_prob_mean=pd.merge(pairs_prob,pairs_prob_mean,left_on="id_window",right_on="id_window")

    pairs_prob_mean=pairs_prob_mean[["window_1", "window_2", "embedding1name","start_window1","end_window1","embedding2name","start_window2","end_window2","id_pair","probability_mean"]]
    pairs_prob_mean.loc[:,"strand1"]="+"
    pairs_prob_mean.loc[:,"strand2"]="+"
    
    pairs_prob_mean = pairs_prob_mean.rename({'probability_mean':'RIME_score'}, axis = 1)
    
    return pairs_prob_mean

def format_output_table_bedpe(df_input):
    df = df_input.copy(deep=True)
    df.loc[:,"window_id1"]=df.loc[:,"window_1"].apply(lambda x: x.split("__")[0])
    df.loc[:,"window_id2"]=df.loc[:,"window_2"].apply(lambda x: x.split("__")[0])
    df.loc[:,"start1"]=df.loc[:,"window_1"].apply(lambda x: x.split("__")[1].split("_")[0])
    df.loc[:,"end1"]=df.loc[:,"window_1"].apply(lambda x: x.split("__")[1].split("_")[1])
    df.loc[:,"start2"]=df.loc[:,"window_2"].apply(lambda x: x.split("__")[1].split("_")[0])
    df.loc[:,"end2"]=df.loc[:,"window_2"].apply(lambda x: x.split("__")[1].split("_")[1])
    return df[["window_id1","start1","end1","window_id2","start2","end2","id_pair","RIME_score","strand1","strand2"]]

def PlotByGene(pairs, gene_x, gene_y, save_path = '', sizes=(15,8)):

    matrix_prob_subset=pairs.copy(deep=True)

    matrix_prob_subset=matrix_prob_subset.loc[matrix_prob_subset["window_1"].str.count(gene_x)>0]
    matrix_prob_subset=matrix_prob_subset.loc[matrix_prob_subset["window_2"].str.count(gene_y)>0]

    matrix_prob_subset.loc[:,"start1"]=matrix_prob_subset.loc[:,"window_1"].apply(lambda x: x.split("__")[0].split("_")[-1])
    matrix_prob_subset.loc[:,"start2"]=matrix_prob_subset.loc[:,"window_2"].apply(lambda x: x.split("__")[0].split("_")[-1])

    matrix_prob_subset.loc[:,"start1"]=matrix_prob_subset.loc[:,"start1"].astype(int)
    matrix_prob_subset.loc[:,"start2"]=matrix_prob_subset.loc[:,"start2"].astype(int)

    matrix_prob_subset=matrix_prob_subset.sort_values(["start1","start2"])


    lista_1=[]
    for i in list(matrix_prob_subset.loc[:,"window_1"]):
        if i not in lista_1:
            lista_1+=[i]

    lista_2=[]
    for i in list(matrix_prob_subset.loc[:,"window_2"]):
        if i not in lista_2:
            lista_2+=[i]


    categoria1 = pd.CategoricalDtype(lista_1,ordered=True)
    categoria2 = pd.CategoricalDtype(lista_2,ordered=True)

    matrix_prob_subset.loc[:,"window_1"]=matrix_prob_subset.loc[:,"window_1"].astype(categoria1)
    matrix_prob_subset.loc[:,"window_2"]=matrix_prob_subset.loc[:,"window_2"].astype(categoria2)

    #pairs_mlx.loc[(pairs_mlx["probability"]< 0.55),"probability"]=None

    matrix_prob_subset=matrix_prob_subset.pivot_table(index="window_1",columns="window_2",values="RIME_score",dropna=False)

    plt.figure(figsize=sizes)
    plt.tight_layout()
    sns.heatmap(matrix_prob_subset, cmap='viridis', annot=True, linewidths=0.5,vmin=0.0,vmax=1.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #plt.show()