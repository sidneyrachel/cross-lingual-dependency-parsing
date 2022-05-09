dataset_prefix = 'external_resources/ud-treebanks-v2.9/'

lang_to_set_to_file_mapper = {
    'english': {
        'train': ['UD_English-EWT/en_ewt-ud-train.conllu', 'UD_English-GUM/en_gum-ud-train.conllu'],
        'dev': ['UD_English-EWT/en_ewt-ud-dev.conllu', 'UD_English-GUM/en_gum-ud-dev.conllu'],
        'test': ['UD_English-EWT/en_ewt-ud-test.conllu', 'UD_English-GUM/en_gum-ud-test.conllu']
    },
    'indonesian': {
        'train': ['UD_Indonesian-GSD/id_gsd-ud-train.conllu', 'UD_Indonesian-CSUI/id_csui-ud-train.conllu'],
        'dev': ['UD_Indonesian-GSD/id_gsd-ud-dev.conllu'],
        'test': [
            'UD_Indonesian-GSD/id_gsd-ud-test.conllu',
            'UD_Indonesian-CSUI/id_csui-ud-test.conllu',
            'UD_Indonesian-PUD/id_pud-ud-test.conllu'
        ]
    },
    'portuguese': {
        'train': ['UD_Portuguese-Bosque/pt_bosque-ud-train.conllu'],
        'dev': ['UD_Portuguese-Bosque/pt_bosque-ud-dev.conllu'],
        'test': ['UD_Portuguese-Bosque/pt_bosque-ud-test.conllu']
    },
    'tagalog': {
        'train': [],
        'dev': [],
        'test': ['UD_Tagalog-TRG/tl_trg-ud-test.conllu', 'UD_Tagalog-Ugnayan/tl_ugnayan-ud-test.conllu']
    },
    'javanese': {
        'train': [],
        'dev': [],
        'test': ['UD_Javanese-CSUI/jv_csui-ud-test.conllu']
    },
    'latvian': {
        'train': ['UD_Latvian-LVTB/lv_lvtb-ud-train.conllu'],
        'dev': ['UD_Latvian-LVTB/lv_lvtb-ud-dev.conllu'],
        'test': ['UD_Latvian-LVTB/lv_lvtb-ud-test.conllu']
    },
    'lithuanian': {
        'train': ['UD_Lithuanian-ALKSNIS/lt_alksnis-ud-train.conllu', 'UD_Lithuanian-HSE/lt_hse-ud-train.conllu'],
        'dev': ['UD_Lithuanian-ALKSNIS/lt_alksnis-ud-dev.conllu', 'UD_Lithuanian-HSE/lt_hse-ud-dev.conllu'],
        'test': ['UD_Lithuanian-ALKSNIS/lt_alksnis-ud-test.conllu', 'UD_Lithuanian-HSE/lt_hse-ud-test.conllu']
    },
    'irish': {
        'train': ['UD_Irish-IDT/ga_idt-ud-train.conllu'],
        'dev': ['UD_Irish-IDT/ga_idt-ud-dev.conllu'],
        'test': ['UD_Irish-IDT/ga_idt-ud-test.conllu', 'UD_Irish-TwittIrish/ga_twittirish-ud-test.conllu']
    },
    'breton': {
        'train': [],
        'dev': [],
        'test': ['UD_Breton-KEB/br_keb-ud-test.conllu']
    },
    'welsh': {
        'train': ['UD_Welsh-CCG/cy_ccg-ud-train.conllu'],
        'dev': ['UD_Welsh-CCG/cy_ccg-ud-dev.conllu'],
        'test': ['UD_Welsh-CCG/cy_ccg-ud-test.conllu']
    },
    'hindi': {
        'train': ['UD_Hindi-HDTB/hi_hdtb-ud-train.conllu'],
        'dev': ['UD_Hindi-HDTB/hi_hdtb-ud-dev.conllu'],
        'test': ['UD_Hindi-PUD/hi_pud-ud-test.conllu', 'UD_Hindi-HDTB/hi_hdtb-ud-test.conllu']
    },
    'bengali': {
        'train': [],
        'dev': [],
        'test': ['UD_Bengali-BRU/bn_bru-ud-test.conllu']
    },
    'marathi': {
        'train': ['UD_Marathi-UFAL/mr_ufal-ud-train.conllu'],
        'dev': ['UD_Marathi-UFAL/mr_ufal-ud-dev.conllu'],
        'test': ['UD_Marathi-UFAL/mr_ufal-ud-test.conllu']
    },
    'turkish': {
        'train': ['UD_Turkish-Kenet/tr_kenet-ud-train.conllu', 'UD_Turkish-Penn/tr_penn-ud-train.conllu'],
        'dev': ['UD_Turkish-Kenet/tr_kenet-ud-dev.conllu', 'UD_Turkish-Penn/tr_penn-ud-dev.conllu'],
        'test': ['UD_Turkish-Kenet/tr_kenet-ud-test.conllu', 'UD_Turkish-Penn/tr_penn-ud-test.conllu']
    },
    'kazakh': {
        'train': ['UD_Kazakh-KTB/kk_ktb-ud-train.conllu'],
        'dev': [],
        'test': ['UD_Kazakh-KTB/kk_ktb-ud-test.conllu']
    },
    'tatar': {
        'train': [],
        'dev': [],
        'test': ['UD_Tatar-NMCTT/tt_nmctt-ud-test.conllu']
    },
    'estonian': {
        'train': ['UD_Estonian-EDT/et_edt-ud-train.conllu', 'UD_Estonian-EWT/et_ewt-ud-train.conllu'],
        'dev': ['UD_Estonian-EDT/et_edt-ud-dev.conllu', 'UD_Estonian-EWT/et_ewt-ud-dev.conllu'],
        'test': ['UD_Estonian-EDT/et_edt-ud-test.conllu', 'UD_Estonian-EWT/et_ewt-ud-test.conllu']
    },
    'hungarian': {
        'train': ['UD_Hungarian-Szeged/hu_szeged-ud-train.conllu'],
        'dev': ['UD_Hungarian-Szeged/hu_szeged-ud-dev.conllu'],
        'test': ['UD_Hungarian-Szeged/hu_szeged-ud-test.conllu']
    },
    'finnish': {
        'train': ['UD_Finnish-TDT/fi_tdt-ud-train.conllu'],
        'dev': ['UD_Finnish-TDT/fi_tdt-ud-dev.conllu'],
        'test': ['UD_Finnish-TDT/fi_tdt-ud-test.conllu']
    },
    'german': {
        'train': ['UD_German-HDT/de_hdt-ud-train.conllu', 'UD_German-GSD/de_gsd-ud-train.conllu'],
        'dev': ['UD_German-HDT/de_hdt-ud-dev.conllu', 'UD_German-GSD/de_gsd-ud-dev.conllu'],
        'test': ['UD_German-HDT/de_hdt-ud-test.conllu', 'UD_German-GSD/de_gsd-ud-test.conllu']
    },
    'afrikaans': {
        'train': ['UD_Afrikaans-AfriBooms/af_afribooms-ud-train.conllu'],
        'dev': ['UD_Afrikaans-AfriBooms/af_afribooms-ud-dev.conllu'],
        'test': ['UD_Afrikaans-AfriBooms/af_afribooms-ud-test.conllu']
    },
    'lowsaxon': {
        'train': [],
        'dev': [],
        'test': ['UD_Low_Saxon-LSDC/nds_lsdc-ud-test.conllu']
    },
    'dutch': {
        'train': ['UD_Dutch-Alpino/nl_alpino-ud-train.conllu', 'UD_Dutch-LassySmall/nl_lassysmall-ud-train.conllu'],
        'dev': ['UD_Dutch-Alpino/nl_alpino-ud-dev.conllu', 'UD_Dutch-LassySmall/nl_lassysmall-ud-dev.conllu'],
        'test': ['UD_Dutch-Alpino/nl_alpino-ud-test.conllu', 'UD_Dutch-LassySmall/nl_lassysmall-ud-test.conllu']
    },
    'french': {
        'train': ['UD_French-GSD/fr_gsd-ud-train.conllu', 'UD_French-ParTUT/fr_partut-ud-train.conllu'],
        'dev': ['UD_French-GSD/fr_gsd-ud-dev.conllu', 'UD_French-ParTUT/fr_partut-ud-dev.conllu'],
        'test': ['UD_French-GSD/fr_gsd-ud-test.conllu', 'UD_French-ParTUT/fr_partut-ud-test.conllu']
    },
    'galician': {
        'train': ['UD_Galician-CTG/gl_ctg-ud-train.conllu', 'UD_Galician-TreeGal/gl_treegal-ud-train.conllu'],
        'dev': ['UD_Galician-CTG/gl_ctg-ud-dev.conllu'],
        'test': ['UD_Galician-CTG/gl_ctg-ud-test.conllu', 'UD_Galician-TreeGal/gl_treegal-ud-test.conllu']
    },
    'czech': {
        'train': ['UD_Czech-PDT/cs_pdt-ud-train.conllu', 'UD_Czech-CAC/cs_cac-ud-train.conllu'],
        'dev': ['UD_Czech-PDT/cs_pdt-ud-dev.conllu', 'UD_Czech-CAC/cs_cac-ud-dev.conllu'],
        'test': ['UD_Czech-PDT/cs_pdt-ud-test.conllu', 'UD_Czech-CAC/cs_cac-ud-test.conllu']
    },
    'serbian': {
        'train': ['UD_Serbian-SET/sr_set-ud-train.conllu'],
        'dev': ['UD_Serbian-SET/sr_set-ud-dev.conllu'],
        'test': ['UD_Serbian-SET/sr_set-ud-test.conllu']
    },
    'russian': {
        'train': ['UD_Russian-Taiga/ru_taiga-ud-train.conllu', 'UD_Russian-SynTagRus/ru_syntagrus-ud-train.conllu'],
        'dev': ['UD_Russian-Taiga/ru_taiga-ud-dev.conllu', 'UD_Russian-SynTagRus/ru_syntagrus-ud-dev.conllu'],
        'test': ['UD_Russian-Taiga/ru_taiga-ud-test.conllu', 'UD_Russian-SynTagRus/ru_syntagrus-ud-test.conllu']
    }
}