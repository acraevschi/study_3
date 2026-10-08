## toy
df_toy <- read_tsv("/media/data/corpora/morphology/unimorph/toy/toy"
             , col_names = c("lexeme", "form", "cell"))
df_toy_v <- filter(df_toy, str_detect(cell, "^V;")) %>% process_subset()
write_csv(df_toy_v, "../data/toy-v.csv")
## epi
df_epi <- read_tsv("/media/data/corpora/morphology/unimorph/epi/epi"
             , col_names = c("lexeme", "form", "cell"))
df_epi$form <- epi_transliterate(df_epi$form, "epi-Latn")
df_epi_n <- filter(df_epi, str_detect(cell, "^N;")) %>% process_subset()
write_csv(df_epi_n, "../data/toy-n.csv")
