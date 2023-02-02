library(dplyr)
library(ggplot2)
library(data.table)

RAW_DATA_PATH <- "/home/thusitha/work/bigdata/recomendation/data_recomndation/user_behaviour_complete.csv" # nolint
META_DATA_PATH <- "/home/thusitha/work/bigdata/recomendation/data_recomndation/item_metadata.csv" # nolint

# Read the raw data
user_dt <- fread(RAW_DATA_PATH)
meta_dt <- fread(META_DATA_PATH)

user_dt
meta_dt

# calclate basic statistics
user_dt[, .(N_USERS = uniqueN(user_id), N_ITEMS = uniqueN(item_id))]
user_interactions_cnt  <- user_dt[, .N, user_id]

user_interactions_cnt %>% ggplot(aes(x = N)) +
    geom_histogram()

user_dt[, .N, behavior_type]  %>% ggplot(aes(x = behavior_type, y = N)) +
    geom_bar(stat = "identity")


user_dt[, .N, item_category] %>% ggplot(aes(x = N)) +
  geom_histogram() +
  xlim(0, 1000)

user_dt[, date:=sapply(time, function(x) {stringr::str_split(x, " ")[[1]][[1]]}, simplify = T)]
