cd examples/language_model/
bash prepare-wikitext-103.sh
cd ../..


TEXT=examples/language_model/wikitext-103
python fairseq_cli/preprocess.py \
    --only-source \
    --trainpref $TEXT/wiki.train.tokens \
    --validpref $TEXT/wiki.valid.tokens \
    --testpref $TEXT/wiki.test.tokens \
    --destdir data-bin/wikitext-103 \
    --workers 20
