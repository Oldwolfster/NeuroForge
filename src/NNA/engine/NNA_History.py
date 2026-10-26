# NNA_history.py
from datetime import datetime


from src.NNA.utils.enums import RecordLevel
import json                         # NEW


class NNA_history:

    @classmethod
    def record(cls, TRI, batch_id, run_id, lego_manager):
        NNA_history.record_from_config (TRI, batch_id,      run_id, lego_manager)

        #NNA_history.copy_ram_db_to_perm(TRI)
        #add_pytorch_views              (TRI.hyper.db_dsk.conn)


    @classmethod
    def record_from_config(cls, TRI, batch_id, run_id, lego_manager): #TODO shuold this have best_accuracy_epoch
        """Record training results to batch_history table."""
        TRI         . record_finish_time()
        cursor = TRI.hyper.nf_db.conn.cursor()
        # DIAGNOSTIC - remove after debugging
        #config_dict = lego_manager.diagnose_serialization(TRI.config, "TRI.config")
        config_dict = lego_manager.serialize_ANYTHING(TRI.config)
        #print(f"config_dict:{config_dict}")

        #config_dict = lego_manager.serialize_ANYTHING(TRI.config)
        #print(f"config_dict:{config_dict}")
        # Add non-config fields
        results = {
            'status': 'completed',
            'gladiator': TRI.gladiator,
            'arena': TRI.training_data.arena_name,
            'epoch_count': TRI.last_epoch,
            'checkpoint_phase': TRI.checkpoint_phase,
            'checkpoint_epoch': TRI.optimum_epoch,
            'convergence_condition': TRI.converge_cond or 'None',
            'runtime_seconds': TRI.time_seconds,
            'seed': TRI.seed,
            'problem_type': TRI.training_data.problem_type,
            'sample_count': TRI.training_data.sample_count,
            'target_min': TRI.training_data.target_min,
            'target_max': TRI.training_data.target_max,
            'target_min_label': TRI.training_data.target_min_label,
            'target_max_label': TRI.training_data.target_max_label,
            'target_mean': TRI.training_data.target_mean,
            'target_stdev': TRI.training_data.target_stdev,
        }
        results.update(cls.phase_block(TRI, 0, 'train'))  # 0 is train
        results.update(cls.phase_block(TRI, 1, 'validate')) #1 is validate
        results.update(cls.test_block(TRI))
        results.update(cls.replay_block(TRI))

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
        TRI.hyper.nf_db.conn.commit()

    @classmethod
    def replay_block(cls, TRI):
        td = TRI.training_data
        return {
            'record_level'      : TRI.record_level.value,
            'input_count'       : td.input_count,
            'feature_labels'    : json.dumps(td.feature_labels),
            'source_code'       : getattr(td, 'source_code', None),
            'backprop_headers'  : json.dumps(TRI.backprop_headers),
            'scaler_params'     : json.dumps(TRI.scaler_params),
        }


    @classmethod
    def phase_block(cls, TRI, phase, suffix):
        """Per-epoch phase. Empty dict when the phase never ran -- its columns stay NULL."""
        if not TRI.phase_ran(phase): return {}
        return {
            f'mae_{suffix}'          : TRI.lowest_mae      [phase],
            f'mae_epoch_{suffix}'    : TRI.lowest_mae_epoch[phase],
            f'sample_count_{suffix}' : TRI.data[phase].sample_count,
        }


    @classmethod
    def test_block(cls, TRI):
        """Test runs once on the crowned weights -- one measurement, so best and last are the same number."""
        if not TRI.phase_ran(2): return {}                          # 2 means test
        return {
            'mae'               : TRI.last_mae     [2],
            'accuracy'          : TRI.last_accuracy[2],
            'sample_count_test' : TRI.data[2].sample_count,
        }



    @classmethod
    def FAST_COPY(cls, TRI, run_id):
        ram_conn = TRI.hyper.db_ram.conn
        #ram_conn.execute("ATTACH DATABASE ? AS disk_db", (str(ProsjectManager.get_db_path()),))
        try:
            ram_conn.execute("""
                INSERT INTO disk_db.epoch_history
                       (run_id, epoch, phase, correct, wrong, accuracy, mae)
                SELECT  run_id, epoch, phase, correct, wrong, accuracy, mae
                FROM    RecordEpoch
                WHERE   run_id = ?
            """, (run_id,))
            ram_conn.commit()
        finally:
            ram_conn.execute("DETACH DATABASE disk_db")

