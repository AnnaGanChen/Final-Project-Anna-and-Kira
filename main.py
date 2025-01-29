import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


# ייבוא הפונקציות מקובץ ניקוי דאטה
from src.data_cleaning import load_and_display_csv_files
from src.data_cleaning import process_merged_dataframe
from src.data_cleaning import process_times
from src.data_cleaning import calculate_means_and_std
from src.data_cleaning import calculate_correlation_matrix
from src.data_cleaning import create_reward_variables
from src.data_cleaning import calculate_reward_averages
from src.data_cleaning import calculate_std_rewards

def main():
    #קריאת פונקציה שמורידה ומאחדת את הדאטה פריים
    merged_df = load_and_display_csv_files()
    
    if merged_df is not None:
        print("Data successfully loaded and merged.") #בדיקה 
        
        # פוננקציה שעושה ניקוי דאטה ,קריאה לפונקציה לניהול הנתונים אחרי הטעינה
        process_merged_dataframe(merged_df) 
        
        # קריאה לפונקציה לעיבוד זמני החלטה
        decision_time_range_result, endowment_result, indexfordecision = process_times(merged_df)
        
        # הצגת התוצאות
        print(merged_df.head())
        print(list(merged_df.columns))

        # חישוב ממוצעים וסטיות תקן
        decision_time_range_mean, endowment_time_range_mean, decision_time_range_std, endowment_time_range_std = calculate_means_and_std(merged_df)
        print(f"ממוצע טווח ההחלטה: {decision_time_range_mean}")
        print(f"ממוצע טווח זמן התמורה: {endowment_time_range_mean}")
        print(f"סטיית תקן עבור טווח ההחלטה: {decision_time_range_std}")
        print(f"סטיית תקן עבור טווח זמן התמורה: {endowment_time_range_std}")

        # חישוב מטריצת קורלציה
        columns_to_select = ['decision_time_range_for_all', 'endowment_time_range_for_all', 'resp_onset']
        correlation_matrix_for_all = calculate_correlation_matrix(merged_df, columns_to_select)
        if correlation_matrix_for_all is not None:
            print(correlation_matrix_for_all)

        # יצירת משתנים חדשים
        merged_df = create_reward_variables(merged_df)
        if merged_df is not None:
            print("Updated DataFrame:")
            print(merged_df.head())

        # חישוב ממוצעי תגמול
        mean_financial_reward, mean_social_reward = calculate_reward_averages(merged_df)
        calculate_std_rewards(merged_df)
        
    else:
        print("There was an error with loading or merging the data.")

    if __name__ == "__main__":
       main()

#time series are in jupyter only for presentation 

from src.statistics import (load_data, calculate_aq_score, categorize_and_group, process_altman_data, 
                            process_aadis_data, group_aadis_by_sum, process_bdi_groups, process_bdi_data, 
                            process_bpaq_data, preprocess_ctqsf, calculate_ctqsf_sums, process_dudit_data, 
                            process_pnr_data, process_pvss_data, process_quic_data, process_self_esteem_data, 
                            process_spsrq_data, process_susd_data, reverse_score_questions, process_and_group_by_mean, 
                            process_teps_data, process_teps_groups_by_threshold, group_tei_by_score, process_tei_scores, 
                            merge_dataframes, classify_participants_by_clinical_occurrences, generate_reward_stats_and_perform_ttest)

def main():
    folder_path = r"phenotype"  #תיקית הבסיס 
    try:
        # להעביר רשימות מהקבצים בתיקיה 
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            # אם הקובץ נמצא בתיקיה - להוריד אותה 
            if file.endswith(".tsv"):
                print(f"Loading {file}...")
                df = load_data(file_path)
                if df is not None:
                    print(df.head())  # להציג 5 שורות ראשונות 
    except Exception as e:
        print(f"Error processing folder {folder_path}: {e}") #בדיקה 

    # AQ questionnaire
    file_path = r'phenotype/autism_quotient.tsv'
    dfaq = load_data(file_path)  # להוריד קובץ 

    if dfaq is not None:
        result = calculate_aq_score(dfaq)  # חישוב ניקוד הסופי, שיטת ניקוד אוניברסלית פר שאלון 
        if result is not None:
            print(result.head())

    grouped_statistics = categorize_and_group(dfaq) #לסווג לקבוצות בלהתבסס למעל או מתחת ל25 מתוך המאמר 
    print(grouped_statistics)

    # Altman self-rating mania scale
    file_path_altman = r'phenotype/altman_self_rating_mania_scale.tsv' #להתייחס לקובץ  ספציפי של השאלון 
    dfmania = load_data(file_path_altman)
    if dfmania is not None:
        print("Data successfully loaded.")
        print(dfmania.columns)

        # עיבוד נתונים של השאלון 
        answers_altman = ["asrm_q1", "asrm_q2", "asrm_q3", "asrm_q4", "asrm_q5"]
        result_mania_sum, grouped, results_altman_final = process_altman_data(dfmania, answers_altman)

        print(result_mania_sum.head())
        print(grouped)
        print(results_altman_final.head())
    else:
        print("There was an error with loading or processing the data.")

    # AADIS questionnaire
    dfaadispath = r'phenotype/adolescent_alcohol_and_drug_involvement_scale.tsv'
    dfaadis = load_data(dfaadispath)

    if dfaadis is not None:
        aadis_sum = process_aadis_data(dfaadis)  # לסכום את כל התוצאות של התשובות לפני חלוקה של קליני/לא קליני
        print("Answers_aadis_sum:\n", aadis_sum.head())

        grouped_aadis = group_aadis_by_sum(dfaadis)  # חלוקה לקליני/לא קליני 
        print("Grouped by sum:\n", grouped_aadis)
    else:
        print("Can't load data")

    # BDI שאלון BECKS DEPRESSION 
    dfbdipath = r'phenotype/becks_depression_inventory.tsv'
    result_bdi = process_bdi_data(dfbdipath)  # עיבוד לפני החלוקה קליני/לא קליני 

    if result_bdi is not None:
        print(result_bdi)

    grouped_bdi, final_table_bdi = process_bdi_groups(result_bdi) #חלוקה לקליני/לא קליני 
    print(grouped_bdi)
    print(final_table_bdi)

    # Behavioral inhibition/activation scale (BIS/BAS)
    dfbisbaspath = r'phenotype/behavioral_inhibition_scale_behavioral_activation_scale.tsv'
    dfbisbas = load_data(dfbisbaspath)
    
    if dfbisbas is not None:
        print(dfbisbas.head()) #עשינו רק משיכה של הפונקציה של דאטה פריים בגלל שבסוף לא השתמשנו בזה 

    #BUSS PERRY AGRESSION שאלון 
    dfbpaqpath = r'phenotype/buss_perry_aggression_questionnaire.tsv'
    dfbpaq = load_data(dfbpaqpath)

    reverse_columns = ["bpaq_q7", "bpaq_q18"]  #שאלות הפוכות 
    reverse_score_bpaq = ["bpaq_q7", "bpaq_q18"]  
    bpaq_answers = [
        "bpaq_q1", "bpaq_q2", "bpaq_q3", "bpaq_q4", "bpaq_q5", "bpaq_q6", "bpaq_q7", "bpaq_q8", "bpaq_q9", "bpaq_q10",
        "bpaq_q11", "bpaq_q12", "bpaq_q13", "bpaq_q14", "bpaq_q15", "bpaq_q16", "bpaq_q17", "bpaq_q18", "bpaq_q19", 
        "bpaq_q20", "bpaq_q21", "bpaq_q22", "bpaq_q23", "bpaq_q24", "bpaq_q25", "bpaq_q26", "bpaq_q27", "bpaq_q28", 
        "bpaq_q29"]  # העמודות למימוש לאחר מכן 

    final_table, grouped_bpaq = process_bpaq_data(dfbpaq, reverse_columns, reverse_score_bpaq, bpaq_answers)
    print(final_table)
    print(grouped_bpaq) #כולל חלוקה לקליני/לא קליני

    #שאלון CHILDHOOD TRAUMA
    dfctqsfpath = r'phenotype/childhood_trauma_questionnaire_short_form.tsv'
    dfctqsf = load_data(dfctqsfpath)

    if dfctqsf is not None:
        print(dfctqsf.head())

        reverse_columns = ["ctqsf_adult_cj_2"]
        denial_columns = ["ctqsf_adult_cj_10", "ctqsf_adult_cj_16", "ctqsf_adult_cj_22"]
        all_answers_columns = [
            "ctqsf_adult_cj_1", "ctqsf_adult_cj_2", "ctqsf_adult_cj_3", "ctqsf_adult_cj_4", "ctqsf_adult_cj_5",
            "ctqsf_adult_cj_6", "ctqsf_adult_cj_7", "ctqsf_adult_cj_8", "ctqsf_adult_cj_9", "ctqsf_adult_cj_11",
            "ctqsf_adult_cj_12", "ctqsf_adult_cj_13", "ctqsf_adult_cj_14", "ctqsf_adult_cj_15", "ctqsf_adult_cj_17",
            "ctqsf_adult_cj_18", "ctqsf_adult_cj_19", "ctqsf_adult_cj_20", "ctqsf_adult_cj_21", "ctqsf_adult_cj_23",
            "ctqsf_adult_cj_24", "ctqsf_adult_cj_25", "ctqsf_adult_cj_26", "ctqsf_adult_cj_27", "ctqsf_adult_cj_28"
        ]

        # מימוש 
        dfctqsf = preprocess_ctqsf(dfctqsf, reverse_columns, denial_columns, all_answers_columns) #חלוקה של הניקוד 
        result = calculate_ctqsf_sums(dfctqsf, all_answers_columns)#חלוקה לקבוצות קלני/לא קליני
        print(result)

    #DRUG USE DISORDER שאלון 
    dfduditfpath = r'phenotype/drug_use_disorders_identification_test.tsv'
    dfdudit = load_data(dfduditfpath)

    if dfdudit is not None:
        print(dfdudit.head())
        result_dudit_sum, grouped_drug_dependance, Drug_dependancy_table = process_dudit_data(dfdudit)
        print(result_dudit_sum.head())
        print(grouped_drug_dependance)
        print(Drug_dependancy_table.head())

    #personal norms of reciprocity שאלון 
    dfpnrfpath = r'phenotype/personal_norms_of_reciprocity.tsv'
    result_pnr_sum, mean_pnr, grouped_negative_r, Reciprocity_table = process_pnr_data(dfpnrfpath) #כולל חלוקה לקליני/לא קליני

    if result_pnr_sum is not None:
        print("PNR Result Sum:")
        print(result_pnr_sum.head())
        print("\nPNR Mean:", mean_pnr)
        print("\nGrouped Negative Reciprocity:")
        print(grouped_negative_r)
        print("\nReciprocity Table:")
        print(Reciprocity_table.head())

    #positive valence system survey שאלון 
    dfpvssfpath = r'phenotype/positive_valence_systems_survey.tsv'
    result_pvss_sum, mean_pvss = process_pvss_data(dfpvssfpath) #כולל חלוקה לקליני/לא קליני

    if result_pvss_sum is not None and mean_pvss is not None:
        print("PVSS Result Sum:")
        print(result_pvss_sum.head())
        print("\nPVSS Mean:", mean_pvss)
    else:
        print("Data didn't process")

    #שאלון unpredictability in childhood
    dfquicfpath = r'phenotype/questionnaire_of_unpredictability_in_childhood.tsv'
    result_quic = process_quic_data(dfquicfpath) #כולל חלוקה לקליני/לא קליני 
    if result_quic is not None:
        print(result_quic.head())

    #R. self esteem שאלון 
    dfsesfpath = r'phenotype/self_esteem_scale.tsv'
    dfself_esteem = load_data(dfsesfpath)
    if dfself_esteem is not None:
        print(dfself_esteem.head())
        self_esteem_result = process_self_esteem_data(dfself_esteem) #כולל חלוקה לקליני/לא קליני
        print(self_esteem_result.head())


    #שאלון seven up/down
    dfsusfpath = r'phenotype/seven_up_seven_down.tsv'
    dfsus = load_data(dfsusfpath)
    if dfsus is not None:
        print(dfsus.head())
        susd_result = process_susd_data(dfsus)
        print(susd_result.head())

    #social experience questionnaire שאלון 
    dfseqfpath = r'phenotype/social_experience_questionnaire.tsv'
    dfseq = load_data(dfseqfpath)
    if dfseq is not None:
        print(dfseq.head())
        susd_result = process_and_group_by_mean(dfseq)
        print(susd_result.head())

    #temporal experience of pleasure שאלון 
    dftepspath = r'phenotype/temporal_experience_of_pleasure_scale.tsv'
    dfteps = load_data(dftepspath)
    if dfteps is not None:
        print(dfteps.head())
        teps_result = process_teps_data(dfteps)
        print(teps_result.head())

    #trait emotional intelligence שאלון 
    dfteifpath = r'phenotype/trait_emotional_intelligence.tsv'
    dftei = load_data(dfteifpath)
    if dftei is not None:
        print(dftei.head())
        tei_result = process_tei_scores(dftei)  # Correctly added process_tei_scores here
        print(tei_result.head())


    # חיבור דאטה פריים 
    all_data_frames = [df, dfaq, dfmania, dfaadis, dfbdipath, dfbisbas, dfbpaq, dfctqsf, dfpnrfpath, dfpvssfpath, dfquicfpath]
    merged_data = merge_dataframes(all_data_frames)
    print(merged_data.head())
    
    # שימוש בהיפוך שאלות 
    reverse_scores = reverse_score_questions(df)
    print(reverse_scores)

    # חישוב ממוצע 
    mean_grouped_data = process_and_group_by_mean(df)
    print(mean_grouped_data)

    # חישוב סף 
    teps_groups = process_teps_groups_by_threshold(dfteps)
    print(teps_groups)

    # לחלק לפי ניקוד 
    tei_grouped = group_tei_by_score(dftei)
    print(tei_grouped)

    # חלוקה לקליני/לא קליני 
    classified_participants = classify_participants_by_clinical_occurrences(df)
    print(classified_participants)

    # טי טסט שלנו 
    reward_stats, ttest_results = generate_reward_stats_and_perform_ttest(df)
    print("Reward Stats:")
    print(reward_stats)
    print("T-test Results:")
    print(ttest_results)
    if __name__ == "__main__":
            main()




