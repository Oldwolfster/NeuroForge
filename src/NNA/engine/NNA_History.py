# NNA_history.py
from datetime import datetime

from src.NNA.utils.db_prep_disk import add_pytorch_views
from src.NNA.utils.enums import RecordLevel


class NNA_history:

    @classmethod
    def record(cls, TRI, batch_id, run_id, lego_manager):
        NNA_history.record_from_config (TRI, batch_id,      run_id, lego_manager)
        NNA_history.copy_ram_db_to_perm(TRI)
        add_pytorch_views              (TRI.hyper.db_dsk.conn)

    @classmethod
    def copy_ram_db_to_perm(cls,TRI): #TRI.hyper.db_ram,   TRI.hyper.db_dsk.conn
        if not TRI.should_record(RecordLevel.FULL)  : return
        if not TRI.run_id==1                        : return
        target                                      = TRI.hyper.db_dsk.conn
        source                                      = TRI.hyper.db_ram
        source.copy_tables_to_permanent             ( target)

    @classmethod
    def record_from_config(cls, TRI, batch_id, run_id, lego_manager):
        """Record training results to batch_history table."""
        TRI         . record_finish_time()
        cursor      = TRI.hyper.db_dsk.conn.cursor()
        # DIAGNOSTIC - remove after debugging
        config_dict = lego_manager.diagnose_serialization(TRI.config, "TRI.config")
        # Was: config_dict = lego_manager.serialize_ANYTHING(TRI.config)
        # Was: print(f"config_dict:{config_dict}")

        #config_dict = lego_manager.serialize_ANYTHING(TRI.config)
        #print(f"config_dict:{config_dict}")
        # Add non-config fields
        results = {
            'status'                : 'completed',
            'gladiator'             : TRI.gladiator,
            'arena'                 : TRI.training_data.arena_name,
            'accuracy'              : TRI.best_accuracy,
            'best_mae'              : TRI.lowest_mae,
            'final_mae'             : TRI.last_mae,
            'epoch_count'           : TRI.last_epoch,
            'convergence_condition' : TRI.converge_cond or 'None',
            'runtime_seconds'       : TRI.time_seconds,
            'seed'                  : TRI.seed,
            'problem_type'          : TRI.training_data.problem_type,
            'sample_count'          : TRI.training_data.sample_count,
            'target_min'            : TRI.training_data.target_min,
            'target_max'            : TRI.training_data.target_max,
            'target_min_label'      : TRI.training_data.target_min_label,
            'target_max_label'      : TRI.training_data.target_max_label,
            'target_mean'           : TRI.training_data.target_mean,
            'target_stdev'          : TRI.training_data.target_stdev,
        }

        all_fields = {**config_dict, **results}     # Merge config and results

        # Build UPDATE statement (row already exists from batch creation)
        set_clause = ', '.join([f"{key} = ?" for key in all_fields.keys()])
        values = list(all_fields.values()) + [run_id, batch_id]

        sql=f'''
            UPDATE batch_history 
            SET {set_clause}
            WHERE run_id = ? AND batch_id = ?
        '''
        print(f"SQL query: {sql}")
        print(f"run id: {run_id} and batch_id: {batch_id}")
        cursor.execute(sql, values)
        TRI.hyper.db_dsk.conn.commit()