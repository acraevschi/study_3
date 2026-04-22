using JudiLing
using CSV
using DataFrames

const INPUT_DIR = "judiling_input"
const OUTPUT_DIR = "judiling_output"
mkpath(OUTPUT_DIR)

const FEATURE_COLS = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10"]

function process_file(lang, sample, split_num, file_path)
    df = DataFrame(CSV.File(file_path, stringtype=String))
    
    # Clean up missing features AND ensure everything is a String
    for col in FEATURE_COLS
        df[!, col] = string.(coalesce.(df[!, col], ""))
    end
    
    out_lang_dir = joinpath(OUTPUT_DIR, lang)
    mkpath(out_lang_dir)

    for fold in 0:3
        train_df = df[df.Fold .!= fold, :]
        val_df = df[df.Fold .== fold, :]

        if nrow(train_df) == 0 || nrow(val_df) == 0
            continue
        end

        # 1. Cue Matrices
        cue_obj_train, cue_obj_val = JudiLing.make_combined_cue_matrix(
            train_df,
            val_df,
            grams = 3,
            target_col = :phonemic_form,
            tokenized = true,
            sep_token = " ",
            keep_sep = true # CHANGED: Set to true to preserve spaces for proper path finding
        )

        # 2. Semantic Matrices
        n_features = size(cue_obj_train.C, 2)
        S_train, S_val = JudiLing.make_combined_S_matrix(
            train_df,
            val_df,
            ["lemma"],
            FEATURE_COLS,
            ncol = n_features
        )

        # 3. Transform Matrices
        G_train = JudiLing.make_transform_matrix(S_train, cue_obj_train.C)
        F_train = JudiLing.make_transform_matrix(cue_obj_train.C, S_train)

        Chat_val = S_val * G_train
        Shat_val = cue_obj_val.C * F_train

        # 4. Learning Paths
        A = cue_obj_train.A
        
        # CHANGED: Added tokenized and sep_token arguments
        max_t = JudiLing.cal_max_timestep(
            train_df, 
            val_df, 
            :phonemic_form,
            tokenized = true,
            sep_token = " "
        )

        res_learn_val, gpi_learn_val = JudiLing.learn_paths(
            train_df,
            val_df,
            cue_obj_train.C,
            S_val,
            F_train,
            Chat_val,
            A,
            cue_obj_train.i2f,
            cue_obj_train.f2i,
            gold_ind = cue_obj_val.gold_ind,
            Shat_val = Shat_val,
            check_gold_path = true,
            max_t = max_t,
            max_can = 10,
            grams = 3,
            threshold = 0.01,
            # is_tolerant = true,
            is_tolerant = false,
            tolerance = -0.1,
            max_tolerance = 2,
            tokenized = true,
            sep_token = " ",           
            keep_sep = true, # CHANGED: Set to true to match cue matrix
            target_col = :phonemic_form,
            # issparse = :dense,
            issparse = :auto,
            verbose = false
        )

        # CHANGED: Fallback to build_paths for any words that learn_paths failed to complete
        missing_indices = findall(x -> length(x) == 0, res_learn_val)
        
        if !isempty(missing_indices)
            val_df_missing = val_df[missing_indices, :]
            S_val_missing = S_val[missing_indices, :]
            Chat_val_missing = Chat_val[missing_indices, :]
            
            res_build = JudiLing.build_paths(
                val_df_missing,
                cue_obj_train.C,
                S_val_missing,
                F_train,
                Chat_val_missing,
                A,
                cue_obj_train.i2f,
                cue_obj_train.gold_ind,
                max_t = max_t,
                max_can = 10,
                n_neighbors = 10,
                grams = 3,
                tokenized = true,
                sep_token = " ",
                target_col = :phonemic_form,
                verbose = false
            )
            
            # Merge the recovered paths back into the original results
            for (idx, res) in zip(missing_indices, res_build)
                res_learn_val[idx] = res
            end
        end

        # 5. Extract Predictions into a DataFrame
        df_pred = JudiLing.write2df(
            res_learn_val,
            val_df,
            cue_obj_train,
            cue_obj_val,
            grams = 3,
            tokenized = true,
            sep_token = " ",           
            start_end_token = "#",
            output_sep_token = " ",    
            path_sep_token = "",
            target_col = :phonemic_form 
        )
        
        # Filter down to just the top prediction per word to align perfectly with val_df
        df_pred_top = df_pred[ismissing.(df_pred.isbest) .| (df_pred.isbest .== true), :]
        
        # Merge target/predictions with lemma to easily calculate NED later
        df_out = DataFrame(
            lemma = val_df.lemma,
            paradigm_slot = val_df.paradigm_slot,
            target = val_df.phonemic_form,
            prediction = df_pred_top.pred 
        )
        
        out_file = joinpath(out_lang_dir, "preds_samp$(sample)_split$(split_num)_fold$(fold).csv")
        CSV.write(out_file, df_out)
    end
end

function main()
    for lang in readdir(INPUT_DIR)
        lang_dir = joinpath(INPUT_DIR, lang)
        isdir(lang_dir) || continue
        
        println("Processing $lang...")
        for file in readdir(lang_dir)
            m = match(r"data_samp(\d+)_split(\d+)\.csv", file)
            if m !== nothing
                process_file(lang, m.captures[1], m.captures[2], joinpath(lang_dir, file))
            end
        end
    end
    println("Done processing all datasets!")
end

main()