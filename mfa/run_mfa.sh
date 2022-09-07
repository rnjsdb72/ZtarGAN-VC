DIR=kss
mfa train_g2p --clean ../lexicon/p_lexicon.txt korean.zip
mfa g2p --clean korean.zip ../raw_data/${DIR} korean.txt
mfa train --clean ../raw_data/${DIR} korean.txt ../preprocessed_data/${DIR}/TextGrid