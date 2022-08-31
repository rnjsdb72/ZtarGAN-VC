DIR=kss
mfa train_g2p ../lexicon/p_lexicon.txt korean.zip
mfa g2p korean.zip ../raw_data/${DIR} korean.txt
mfa train ../raw_data/${DIR} korean.txt ../preprocessed_data/${DIR}/TextGrid