import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
import numpy as np
from collections import defaultdict
import argparse
import csv
import scipy.stats as stats



def main(args):
    embeddings_file_path = args.embeddings
    data_dir = args.data_dir
    csv_filename = args.csv_path
    csv2_filename = args.csv2_path


    with open(embeddings_file_path, 'rb') as emb_file:
        embeddings_dict = pickle.load(emb_file)

    with open(csv2_filename, 'w', newline='') as csv2_file:
        csv2_writer = csv.writer(csv2_file)
        csv2_writer.writerow(['filename', 'f1_score_macro', 'all_or_nothing_score'])  
        with open(csv_filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['file_name', 'f1_score_macro']) 
                           
            props_f1_dict = defaultdict(list)
            props_all_or_nthg_dict = defaultdict(list)
            for folder_name in os.listdir(data_dir):   # instance folder
                folder_path = os.path.join(data_dir, folder_name)
                n=3
                m=7
                true_labels = [1] * n + [0] * m 

                total_files = 0
                for file_name in os.listdir(folder_path): # prop in instance
                    file_path = os.path.join(folder_path, file_name)
                    if os.path.isfile(file_path):
                        with open(file_path, 'r') as file:
                            lines = file.readlines()
                            if len(lines) != n + m + 1:
                                continue
                            instance = [line.strip() for line in lines if line.strip()]
                            print('true line: ', instance)
                            line_embeddings = [embeddings_dict[concept] for concept in instance]
                            
                            lowest_score = 100000
                            pos_list = []
                            for i, embedding in enumerate(line_embeddings):
                                other_embeddings = line_embeddings[:i] + line_embeddings[i+1:]
                                id_of_other_embeddings = list(range(i)) + list(range(i + 1, len(line_embeddings)))
                                #print ('i : ', i, 'id_of_other_embeddings : ',id_of_other_embeddings, '\n')
                                cos_similarities = cosine_similarity([embedding], other_embeddings)[0]

                                if args.zscore:
                                    zscores = stats.zscore(cos_similarities)
                                    sorted_indices_by_z_score = np.argsort(zscores)[::-1]
                                    sorted_z_scores = np.array(zscores)[sorted_indices_by_z_score]
                                    #print('sorted_z_scores: ', sorted_z_scores)
                                    sorted_indices = np.array(id_of_other_embeddings)[sorted_indices_by_z_score]
                                    #print('\n sorted_indices: ', sorted_indices)                             
                                    most_similar_lines_indices = sorted_indices[:n-1]
                                    most_dissimilar_lines_indices = sorted_indices[n-1:]
                                    #print('most_similar_lines_indices: ',most_similar_lines_indices) 
                                    pos_list_current = [instance[k] for k in most_similar_lines_indices]
                                    pos_list_current.append(instance[i])  # Including the current line in the positive list
                                    #print('pos_list_current: ',pos_list_current)
                                    max_similar_dist = sorted_z_scores[n-2]
                                    min_dissimilar_dist = sorted_z_scores[n-1]
                                    #print('max_similar_dist: ',  max_similar_dist, 'min_dissimilar_dist: ',min_dissimilar_dist)
                                    
                                else :
                                    sorted_cos = cos_similarities.argsort()[::-1]
                                    sorted_indices = [id_of_other_embeddings[idx] for idx in cos_similarities.argsort()[::-1]]
                                    #print('\n sorted_cos: ', sorted_cos)
                                    #print('\n sorted_indices: ', sorted_indices)                              
                                    most_similar_lines_indices = sorted_indices[:n-1]
                                    most_dissimilar_lines_indices = sorted_indices[n-1:]
                                    #print('most_similar_lines_indices: ',most_similar_lines_indices)            
                                    pos_list_current = [instance[k] for k in most_similar_lines_indices]                                    
                                    pos_list_current.append(instance[i])  # Including the current line in the positive list
                                    #print('pos_list_current: ',pos_list_current)                            
                                    max_similar_dist = sorted_cos[n-2]
                                    min_dissimilar_dist = sorted_cos[n-1]
                                
                                if args.method=='minus':
                                    score = max_similar_dist - min_dissimilar_dist
                                else:
                                    score = max_similar_dist / min_dissimilar_dist
                                
                                if score < lowest_score:
                                    lowest_score = score
                                    pos_list = [instance[k] for k in most_similar_lines_indices]
                                    pos_list.append(instance[i])
                        
                            pred_labels = [1 if concept in pos_list else 0 for concept in instance]
                            if  pred_labels == true_labels:
                                all_or_nothing_score = 1 
                            else:
                                all_or_nothing_score = 0

                            print('pred_labels: ', pred_labels)
                            total_files += 1


                            f1 = f1_score(true_labels, pred_labels, average='macro', pos_label=1)
                            csv_writer.writerow([file_name, f"{f1 * 100:.2f}"])
                            props_f1_dict[file_name].append(f1)
                            props_all_or_nthg_dict[file_name].append(all_or_nothing_score)
                                
        
            for key, values in props_f1_dict.items():
                if values: # Check if the list of values is not empty
                    avg_f1 = np.mean(values) * 100  # Calculate the macro average and convert to percentage
                    macro_avg_f1 = round(avg_f1, 2)
                    csv2_writer.writerow(['f1', key, macro_avg_f1])

            for key, values in props_all_or_nthg_dict.items():
                if values: 
                    avg = np.mean(values) * 100  
                    csv2_writer.writerow(['all_nthg', key, avg])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process positive and negative word embeddings.")
    parser.add_argument("embeddings", help="Number of positives")
    parser.add_argument("data_dir", help="Number of negatives")
    parser.add_argument("csv_path", help="path to save csv files")
    parser.add_argument("csv2_path", help="path to save csv files")
    parser.add_argument("method", help="minus or division")
    parser.add_argument('--zscore', action='store_true', help='Enable zscore.')

    args = parser.parse_args()
    main(args)
    
    


