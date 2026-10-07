"""Reuse the fixed unit-bootstrap calculation for the raw x support comparison."""
import cnh_training_support_contrasts_dev as C


def main():
    C.OUT = C.I.R.WORK / 'cnh-raw-training-support-dev-20261007'
    C.PAIRS = [('raw_source_repeat', 'raw_current'),
               ('raw_hb_aug', 'raw_source_repeat'),
               ('raw_hb_aug', 'raw_current'),
               ('raw_hb_aug', 'pair_hb_aug'),
               ('raw_source_repeat', 'pair_source_repeat'),
               ('raw_hb_aug', 'original_center')]
    C.main()


if __name__ == '__main__':
    main()
