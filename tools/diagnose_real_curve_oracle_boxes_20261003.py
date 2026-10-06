"""GT-box coordinate substitution only; explicitly not model performance."""
import evaluate_real_curve_transfer_20261003 as transfer
transfer.OUT=transfer.CYCLE/'real_curve_oracle_box_v1'
transfer.ORACLE_DIAGNOSTIC=True
if __name__=='__main__':
    try:transfer.main()
    except Exception:
        transfer.OUT.mkdir(exist_ok=True);transfer.write_json(transfer.OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
