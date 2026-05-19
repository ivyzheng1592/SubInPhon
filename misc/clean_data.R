# package
library(fs)
library(here)
library(readxl)
library(writexl)
library(tidyverse)
library(data.table)

####################all acc####################
trials = c("EnglishBH_txt_v", "EnglishBH_fea_v",
           "EnglishBH_nonidentical_txt_v", "EnglishBH_nonidentical_fea_v",
           "EnglishBH_expanded_txt_v", "EnglishBH_expanded_fea_v",
           "EnglishBH_txt_cv", "EnglishBH_fea_cv")

# read accuracy data
acc <- tibble()
for (trial in trials) {
  acc_file_list <- 
    list.files(trial, pattern = "acc\\.csv$", full.names = TRUE)
  for (file in acc_file_list) {
    this_run <- 
      read_csv(here(file)) %>% 
      mutate(trial = trial)
    acc <- bind_rows(this_run, acc)
    rm(this_run)
  }
}

# get the list of all runs where test accuracy did not reach 90% 
# or loss did not reach 0.02
failed_run_list <- 
  acc %>% 
  filter(epoch == 99,
         record_type == "test",
         acc < 0.85 | loss > 0.05) %>% 
  distinct(language, modality, directionality, 
           property, condition, run_num, 
           epoch, record_type, .keep_all = TRUE)

# clean accuracy data
acc <- 
  anti_join(acc, failed_run_list, 
            by = c("language", "modality", "directionality",
                   "property", "condition", "run_num")) %>% 
  mutate(model = if_else(str_detect(modality, "txt"),
                         "segment", "feature"),
         directionality = if_else(directionality == "l2r",
                                "left-to-right", "right-to-left"),
         dataset = case_when(str_detect(language, "expanded") ~ "expanded",
                             is.na(property) ~ "full",
                             property == "nonidentical" ~ "reduced"),
         error_record = if_else(str_detect(trial, "cv"),
                                "cv", "v")) %>% 
  rename(subset = record_type) %>% 
  select(language, model, directionality, dataset, error_record, 
         condition, run_num, epoch:acc)
acc

# get the list of all included runs
run_list <- 
  acc %>% 
  distinct(language, model, directionality, dataset, error_record,
           condition, run_num, .keep_all = TRUE) %>%
  group_by(language, model, directionality, dataset, error_record, 
           condition) %>% 
  summarise(n = n())
run_list

# save cleaned data to file
write_csv(acc, "cleaned_260518_EnglishBH_all_acc.csv")
rm(list = ls())
gc()

####################cv pred####################
trials = c("EnglishBH_txt_cv", "EnglishBH_fea_cv")

# read accuracy data
acc_list <- list()
for (trial in trials) {
  acc_file_list <- 
    list.files(trial, pattern = "acc\\.csv$", full.names = TRUE)
  for (file in acc_file_list) {
    this_run <- read_csv(here(file))
    acc_list[[length(acc_list) + 1]] <- this_run
    rm(this_run)
    gc()
  }
}
acc <- rbindlist(acc_list)
rm(acc_list)
gc()

# get the list of all runs where test accuracy did not reach 90% 
# or loss did not reach 0.02
failed_run_list <- 
  acc %>% 
  filter(epoch == 99,
         record_type == "test",
         acc < 0.85 | loss > 0.05) %>% 
  distinct(language, modality, directionality, 
           property, condition, run_num, 
           epoch, record_type, .keep_all = TRUE)

# clean accuracy data
acc <- 
  # remove failed runs
  anti_join(acc, failed_run_list, 
            by = c("language", "modality", "directionality",
                   "property", "condition", "run_num")) %>% 
  mutate(model = if_else(str_detect(modality, "txt"),
                         "segment", "feature"),
         directionality = if_else(directionality == "l2r",
                                  "left-to-right", "right-to-left"),
         dataset = "full") %>% 
  rename(subset = record_type) %>% 
  mutate(total_data = 167968,
         data_split = case_when(subset == "train" ~ 0.8,
                                subset == "test" ~ 0.1),
         subset_data = total_data * data_split,
         total_error = as.integer(subset_data * (1-acc)))
acc

# read and append prediction data
pred_summary_file <- "cleaned_260518_EnglishBH_cv_pred"

# remove existing file
if(file.exists(pred_summary_file))
  file.remove(pred_summary_file)
first_write <- TRUE

# select needed columns
pred_cols <- c("language", "modality", "directionality", "property",
               "condition", "run_num", "epoch", "record_type",
               "v1_error", "v2_error", "o1_error", "o2_error",
               "c1_error", "c2_error", "pred_sr_v1", "pred_sr_v2")

for (trial in trials) {
  pred_file_list <- 
    list.files(trial, pattern = "pred\\.csv$", full.names = TRUE)
  
  for (file in pred_file_list) {
    this_run <- fread(here(file), select = pred_cols)
    
    this_run <- this_run %>% 
      # remove failed runs
      anti_join(failed_run_list,
                by = c("language", "modality", "directionality", "property", 
                       "condition", "run_num")) %>%
      mutate(model = if_else(str_detect(modality, "txt"),
                             "segment", "feature"),
             directionality = if_else(directionality == "l2r",
                                      "left-to-right", "right-to-left"),
             dataset = "full") %>%
      rename(subset = record_type) %>%
      mutate(v_error = if_else(v1_error == 0 & v2_error == 0,
                               0, 1),
             c_error = if_else(o1_error == 0 & o2_error == 0 &
                                 c1_error == 0 & c2_error == 0,
                               0, 1),
             pred_sr_v1_back = case_when(
               pred_sr_v1 %in% c("i", "ɪ", "e", "ɛ") ~ 0,
               pred_sr_v1 %in% c("u", "ʊ", "o", "ɔ") ~ 1),
             pred_sr_v2_back = case_when(
               pred_sr_v2 %in% c("i", "ɪ", "e", "ɛ") ~ 0,
               pred_sr_v2 %in% c("u", "ʊ", "o", "ɔ") ~ 1),
             harmony_error = case_when(
               condition == "harmony" & pred_sr_v1_back != pred_sr_v2_back ~ 1,
               condition == "disharmony" & pred_sr_v1_back == pred_sr_v2_back ~ 1,
               TRUE ~ 0))
    
    this_summary <- this_run %>% 
      # consonant and vowel errors
      group_by(model, directionality, dataset, 
               condition, run_num, epoch, subset, 
               c_error, v_error) %>% 
      summarise(error_num = n(), .groups = "drop") %>% 
      # add the epochs with no consonant and/or vowel errors
      complete(nesting(model, directionality, dataset, 
                       condition, run_num), 
               nesting(epoch, subset), c_error, v_error, 
               fill = list(error_num = 0)) %>% 
      # syllable structure errors
      group_by(model, directionality, dataset, condition, 
               run_num, epoch, subset) %>% 
      mutate(segment_error = sum(error_num)) %>% 
      left_join(acc, by = c("model", "directionality", "dataset", "condition", 
                            "run_num", "epoch", "subset"),
                keep = FALSE) %>% 
      select(-loss, -acc) %>% 
      mutate(error_num = if_else(c_error == 0 & v_error == 0,
                                 total_error - segment_error,
                                 error_num)) %>% 
      # reshape
      mutate(error_type = case_when(c_error == 1 & v_error == 0 ~ 
                                      "consonant only",
                                    c_error == 0 & v_error == 1 ~ 
                                      "vowel only",
                                    c_error == 1 & v_error == 1 ~ 
                                      "consonant and vowel",
                                    c_error == 0 & v_error == 0 ~ 
                                      "syllable structure"),
             error_type = factor(error_type,
                                 levels = c("syllable structure",
                                            "consonant and vowel",
                                            "consonant only",
                                            "vowel only"))) %>%
      select(-c_error, -v_error, -segment_error) %>% 
      relocate(error_type, .before = error_num) %>% 
      mutate(error_rate = if_else(total_error == 0, 0,
                                  error_num/total_error))
    
    # write to file
    fwrite(this_summary, pred_summary_file,
           append = !first_write, col.names = first_write)
    first_write <- FALSE
    
    rm(this_run, this_summary)
    gc()
  }
}

rm(list = ls())
gc()

####################v pred####################
trials = c("EnglishBH_txt_v", "EnglishBH_fea_v",
           "EnglishBH_nonidentical_txt_v", "EnglishBH_nonidentical_fea_v",
           "EnglishBH_expanded_txt_v", "EnglishBH_expanded_fea_v")

# read accuracy data
acc_list <- list()
for (trial in trials) {
  acc_file_list <- 
    list.files(trial, pattern = "acc\\.csv$", full.names = TRUE)
  for (file in acc_file_list) {
    this_run <- read_csv(here(file))
    acc_list[[length(acc_list) + 1]] <- this_run
    rm(this_run)
    gc()
  }
}
acc <- rbindlist(acc_list)
rm(acc_list)
gc()

# get the list of all runs where test accuracy did not reach 90% 
# or loss did not reach 0.02
failed_run_list <- 
  acc %>% 
  filter(epoch == 99,
         record_type == "test",
         acc < 0.85 | loss > 0.05) %>% 
  distinct(language, modality, directionality, 
           property, condition, run_num, 
           epoch, record_type, .keep_all = TRUE)

# read and append predictions data
pred_summary_file <- "cleaned_260518_EnglishBH_v_pred.csv"
vowel_height_file <- "cleaned_260518_EnglishBH_v_height.csv"

# remove existing file
if (file.exists(pred_summary_file))
  file.remove(pred_summary_file)
if (file.exists(vowel_height_file))
  file.remove(vowel_height_file)
first_pred_write <- TRUE
first_vowel_height_write <- TRUE

# expand columns to adapt to the expanded dataset
pred_cols <- c("language", "modality", "directionality", "property",
               "condition", "run_num", "epoch", "record_type",
               "v1_error", "v2_error", "v3_error",
               "sr_v1", "sr_v2", "sr_v3",
               "pred_sr_v1", "pred_sr_v2", "pred_sr_v3")

for (trial in trials) {
  pred_file_list <- 
    list.files(trial, pattern = "pred\\.csv$", full.names = TRUE)
  
  for (file in pred_file_list) {
    this_run <- fread(here(file))
    
    # append missing columns for full and reduced datasets
    missing_cols <- setdiff(pred_cols, names(this_run))
    this_run[, (missing_cols) := NA]
    
    this_run <- this_run %>% 
      # remove failed runs
      anti_join(failed_run_list, 
                by = c("language", "modality", "directionality",
                       "property", "condition", "run_num")) %>% 
      mutate(model = if_else(str_detect(modality, "txt"),
                             "segment", "feature"),
             directionality = if_else(directionality == "l2r",
                                      "left-to-right", "right-to-left"),
             dataset = case_when(str_detect(language, "expanded") ~ "expanded",
                                 is.na(property) ~ "full",
                                 property == "nonidentical" ~ "reduced")) %>% 
      rename(subset = record_type) %>% 
      mutate(sr_v1_high = case_when(
               sr_v1 %in% c("i", "ɪ", "u", "ʊ") ~ 1,
               sr_v1 %in% c("e", "ɛ", "o", "ɔ") ~ 0),
             sr_v1_tense = case_when(
               sr_v1 %in% c("i", "u", "e", "o") ~ 1,
               sr_v1 %in% c("ɪ", "ʊ", "ɛ", "ɔ") ~ 0),
             sr_v1_back = case_when(
               sr_v1 %in% c("i", "ɪ", "e", "ɛ") ~ 0,
               sr_v1 %in% c("u", "ʊ", "o", "ɔ") ~ 1),
             sr_v2_high = case_when(
               sr_v2 %in% c("i", "ɪ", "u", "ʊ") ~ 1,
               sr_v2 %in% c("e", "ɛ", "o", "ɔ") ~ 0),
             sr_v2_tense = case_when(
               sr_v2 %in% c("i", "u", "e", "o") ~ 1,
               sr_v2 %in% c("ɪ", "ʊ", "ɛ", "ɔ") ~ 0),
             sr_v2_back = case_when(
               sr_v2 %in% c("i", "ɪ", "e", "ɛ") ~ 0,
               sr_v2 %in% c("u", "ʊ", "o", "ɔ") ~ 1),
             sr_v3_high = case_when(
               sr_v3 %in% c("i", "ɪ", "u", "ʊ") ~ 1,
               sr_v3 %in% c("e", "ɛ", "o", "ɔ") ~ 0),
             sr_v3_tense = case_when(
               sr_v3 %in% c("i", "u", "e", "o") ~ 1,
               sr_v3 %in% c("ɪ", "ʊ", "ɛ", "ɔ") ~ 0),
             sr_v3_back = case_when(
               sr_v3 %in% c("i", "ɪ", "e", "ɛ") ~ 0,
               sr_v3 %in% c("u", "ʊ", "o", "ɔ") ~ 1),
             pred_sr_v1_high = case_when(
               pred_sr_v1 %in% c("i", "ɪ", "u", "ʊ") ~ 1,
               pred_sr_v1 %in% c("e", "ɛ", "o", "ɔ") ~ 0),
             pred_sr_v1_tense = case_when(
               pred_sr_v1 %in% c("i", "u", "e", "o") ~ 1,
               pred_sr_v1 %in% c("ɪ", "ʊ", "ɛ", "ɔ") ~ 0),
             pred_sr_v1_back = case_when(
               pred_sr_v1 %in% c("i", "ɪ", "e", "ɛ") ~ 0,
               pred_sr_v1 %in% c("u", "ʊ", "o", "ɔ") ~ 1),
             pred_sr_v2_high = case_when(
               pred_sr_v2 %in% c("i", "ɪ", "u", "ʊ") ~ 1,
               pred_sr_v2 %in% c("e", "ɛ", "o", "ɔ") ~ 0),
             pred_sr_v2_tense = case_when(
               pred_sr_v2 %in% c("i", "u", "e", "o") ~ 1,
               pred_sr_v2 %in% c("ɪ", "ʊ", "ɛ", "ɔ") ~ 0),
             pred_sr_v2_back = case_when(
               pred_sr_v2 %in% c("i", "ɪ", "e", "ɛ") ~ 0,
               pred_sr_v2 %in% c("u", "ʊ", "o", "ɔ") ~ 1),
             pred_sr_v3_high = case_when(
               pred_sr_v3 %in% c("i", "ɪ", "u", "ʊ") ~ 1,
               pred_sr_v3 %in% c("e", "ɛ", "o", "ɔ") ~ 0),
             pred_sr_v3_tense = case_when(
               pred_sr_v3 %in% c("i", "u", "e", "o") ~ 1,
               pred_sr_v3 %in% c("ɪ", "ʊ", "ɛ", "ɔ") ~ 0),
             pred_sr_v3_back = case_when(
               pred_sr_v3 %in% c("i", "ɪ", "e", "ɛ") ~ 0,
               pred_sr_v3 %in% c("u", "ʊ", "o", "ɔ") ~ 1),
             high_error = case_when(
               dataset == "expanded" & 
                 (sr_v1_high != pred_sr_v1_high |
                    sr_v2_high != pred_sr_v2_high |
                    sr_v3_high != pred_sr_v3_high) ~ 1,
               dataset != "expanded" & 
                 (sr_v1_high != pred_sr_v1_high |
                    sr_v2_high != pred_sr_v2_high) ~ 1,
               TRUE ~ 0),
             tense_error = case_when(
               dataset == "expanded" & 
                 (sr_v1_tense != pred_sr_v1_tense |
                    sr_v2_tense != pred_sr_v2_tense |
                    sr_v3_tense != pred_sr_v3_tense) ~ 1,
               dataset != "expanded" & 
                 (sr_v1_tense != pred_sr_v1_tense |
                    sr_v2_tense != pred_sr_v2_tense) ~ 1,
               TRUE ~ 0),
             back_error = case_when(
               dataset == "expanded" & 
                 (sr_v1_back != pred_sr_v1_back |
                    sr_v2_back != pred_sr_v2_back |
                    sr_v3_back != pred_sr_v3_back) ~ 1,
               dataset != "expanded" & 
                 (sr_v1_back != pred_sr_v1_back |
                    sr_v2_back != pred_sr_v2_back) ~ 1,
               TRUE ~ 0),
             harmony_error = case_when(
               dataset == "expanded" & condition == "harmony" & 
                 !paste0(pred_sr_v1_back, pred_sr_v2_back, pred_sr_v3_back) %in% c("000", "111") ~ 1,
               dataset == "expanded" & condition == "disharmony" & 
                 directionality == "left-to-right" &
                 !paste0(pred_sr_v1_back, pred_sr_v2_back, pred_sr_v3_back) %in% c("100", "011") ~ 1,
               dataset == "expanded" & condition == "disharmony" & 
                 directionality == "right-to-left" &
                 !paste0(pred_sr_v1_back, pred_sr_v2_back, pred_sr_v3_back) %in% c("110", "001") ~ 1,
               dataset != "expanded" & condition == "harmony" & 
                 pred_sr_v1_back != pred_sr_v2_back ~ 1,
               dataset != "expanded" & condition == "disharmony" & 
                 pred_sr_v1_back == pred_sr_v2_back ~ 1,
               TRUE ~ 0))
    
    this_summary <- this_run %>% 
      group_by(model, directionality, dataset,
               condition, run_num, epoch, subset,
               v1_error, v2_error, high_error, tense_error, back_error,
               harmony_error) %>% 
      summarise(error_num = n(), .groups = "drop")
    
    this_input_height_summary <- this_run %>%
      filter(dataset == "full", condition == "harmony") %>%
      mutate(v_high = paste0(sr_v1_high, sr_v2_high),
             v_agree = case_when(sr_v1_high == sr_v2_high ~ 1,
                                 sr_v1_high != sr_v2_high ~ 0,
                                 TRUE ~ NA)) %>%
      group_by(model, directionality, dataset, condition, run_num,
               v_high, v_agree) %>%
      summarise(error_num = n(), .groups = "drop") %>%
      # the proportion of each vowel error type out of all vowel errors
      group_by(model, directionality, dataset, condition, run_num) %>%
      mutate(error_rate = error_num/sum(error_num)) %>%
      # add missing values
      ungroup() %>%
      complete(nesting(model, directionality, dataset, condition, run_num),
               nesting(v_high, v_agree),
               fill = list(error_num = 0,
                           error_rate = 0.00))

    this_output_height_summary <- this_run %>%
      filter(dataset == "full", condition == "harmony") %>%
      mutate(v_high = paste0(pred_sr_v1_high, pred_sr_v2_high),
             v_agree = case_when(pred_sr_v1_high == pred_sr_v2_high ~ 1,
                                 pred_sr_v1_high != pred_sr_v2_high ~ 0,
                                 TRUE ~ NA)) %>%
      group_by(model, directionality, dataset, condition, run_num,
               v_high, v_agree) %>%
      summarise(error_num = n(), .groups = "drop") %>%
      # the proportion of each vowel error type out of all vowel errors
      group_by(model, directionality, dataset, condition, run_num) %>%
      mutate(error_rate = error_num/sum(error_num)) %>%
      # add missing values
      ungroup() %>%
      complete(nesting(model, directionality, dataset, condition, run_num),
               nesting(v_high, v_agree),
               fill = list(error_num = 0,
                           error_rate = 0.00))

    this_height_summary <-
      full_join(this_input_height_summary, this_output_height_summary,
                by = c("model", "directionality", "dataset", "condition",
                       "run_num", "v_high", "v_agree"),
                suffix = c("_input", "_output")) %>%
      mutate(error_num_diff = error_num_output - error_num_input,
             error_rate_diff = error_rate_output - error_rate_input)
    
    # write to file
    fwrite(this_summary, pred_summary_file,
           append = !first_pred_write,
           col.names = first_pred_write)
    fwrite(this_height_summary, vowel_height_file,
           append = !first_vowel_height_write,
           col.names = first_vowel_height_write)
    first_pred_write <- FALSE
    first_vowel_height_write <- FALSE
    
    rm(this_run, this_summary, this_height_summary,
       this_input_height_summary, this_output_height_summary)
    gc()
  }
}

rm(list = ls())
gc()
