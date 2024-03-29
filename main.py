def main():
    import os
    from meta.meta import MetaData
    from utils.pretty_output import output_prettifier

    local_save_path = "saved_data/"
    main_bucket_name = 'your-name'
    tmp_bucket_name = 'your-name-tmp'
    logs_path = 'output/'
    target_log_name = 'meta_data_latest.pkl'
    model_path = "model/model_checkpoint/"

    meta_obj = MetaData(logs_path=logs_path,
                        meta_logname=target_log_name,
                        main_bucket_name=main_bucket_name,
                        tmp_bucket_name=tmp_bucket_name,
                        path_to_model=model_path,
                        local_data_save_path=local_save_path)

    meta_obj.preprocessing()
    meta_obj.compare_videos()
    output_prettifier(os.path.join(logs_path, target_log_name))


if __name__ == '__main__':
    from utils.logger import LOGGING_CONFIG
    from logging.config import dictConfig

    # from utils.bug_fixes import fix_download_bug  # for restarting from checkpoint 
    # fix_download_bug()

    dictConfig(LOGGING_CONFIG)
    main()
