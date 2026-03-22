# Distribution Shift Report: Private vs Warmup

## Answer Type Distribution

| Answer Type | Private | Warmup | Delta |
|-------------|---------|--------|-------|
| boolean | 193 | 32 | +161 |
| date | 93 | 1 | +92 |
| free_text | 270 | 30 | +240 |
| name | 95 | 15 | +80 |
| names | 90 | 5 | +85 |
| number | 159 | 17 | +142 |
| **Total** | **900** | **100** | |

## Scope Mode Distribution

| Scope Mode | Private | Warmup | Delta |
|------------|---------|--------|-------|
| broad_free_text | 225 | 10 | +215 |
| compare_pair | 162 | 28 | +134 |
| explicit_page | 2 | 1 | +1 |
| full_case_files | 22 | 4 | +18 |
| negative_unanswerable | 3 | 4 | -1 |
| single_field_single_doc | 486 | 53 | +433 |

## Complexity Distribution

| Complexity Bucket | Count |
|-------------------|-------|
| 0 (trivial) | 302 |
| 1-2 (simple) | 483 |
| 3-5 (moderate) | 115 |
| 6+ (complex) | 0 |

## New Document Families (not seen in warmup)

| Doc ID (prefix) | Title | Detected Type | Warmup Similarity |
|-----------------|-------|---------------|-------------------|
| 00427a0402e7... |  | other | 0.00 |
| 01881c184876... |  | other | 0.00 |
| 0223ebe925fc... |  | other | 0.00 |
| 02a351b5d6db... |  | other | 0.00 |
| 02ec30ff6c98... |  | other | 0.00 |
| 03b621728fe2... |  | other | 0.00 |
| 03b72458c300... |  | other | 0.00 |
| 0428cdda3cb1... |  | other | 0.00 |
| 0471e83c1ea1... |  | other | 0.00 |
| 04be93255ec4... |  | other | 0.00 |
| 0511012a2d66... |  | other | 0.00 |
| 057397024d83... |  | other | 0.00 |
| 07853510e976... |  | other | 0.00 |
| 0a18273af203... |  | other | 0.00 |
| 0a59bb9e8700... |  | other | 0.00 |
| 0bf0e451b14c... |  | other | 0.00 |
| 0c3661cf1a10... |  | other | 0.00 |
| 0c4a738afba1... |  | other | 0.00 |
| 0e67dea643dc... |  | other | 0.00 |
| 0ebf2364f577... |  | other | 0.00 |
| 0ed03973e0da... |  | other | 0.00 |
| 0ef7f18935aa... |  | other | 0.00 |
| 0f979e960d3a... |  | other | 0.00 |
| 0fab7473adef... |  | other | 0.00 |
| 0ffd81e8fd8b... |  | other | 0.00 |
| 103a9b704ee8... |  | other | 0.00 |
| 1048c5836cde... |  | other | 0.00 |
| 108efd7d2665... |  | other | 0.00 |
| 1097d4b46b95... |  | other | 0.00 |
| 11409039171b... |  | other | 0.00 |
| 11dd07e61012... |  | other | 0.00 |
| 13538e4c8db1... |  | other | 0.00 |
| 138dc1cd698c... |  | other | 0.00 |
| 138ff0271bda... |  | other | 0.00 |
| 13f30e764b66... |  | other | 0.00 |
| 13f4f5176e77... |  | other | 0.00 |
| 156c7fec26ed... |  | other | 0.00 |
| 159dcd9e773b... |  | other | 0.00 |
| 1635b7381219... |  | other | 0.00 |
| 18d5d0320cde... |  | other | 0.00 |
| 1ae3f9d65856... |  | other | 0.00 |
| 1d07d30a1793... |  | other | 0.00 |
| 1e6a579d52e9... |  | other | 0.00 |
| 1e6d5be08bab... |  | other | 0.00 |
| 1f78ba7075f9... |  | other | 0.00 |
| 1fbae05f3818... |  | other | 0.00 |
| 2071fcaf840c... |  | other | 0.00 |
| 20be16a68c4d... |  | other | 0.00 |
| 23c73b12ed7c... |  | other | 0.00 |
| 2479d3be6e11... |  | other | 0.00 |
| 265968877f0f... |  | other | 0.00 |
| 26dbc112e53a... |  | other | 0.00 |
| 27f2898ad3be... |  | other | 0.00 |
| 29c70158f176... |  | other | 0.00 |
| 2a2461886df0... |  | other | 0.00 |
| 2a803bd388de... |  | other | 0.00 |
| 2bc070593cb0... |  | other | 0.00 |
| 30f90aeb909b... |  | other | 0.00 |
| 330c4f93e809... |  | other | 0.00 |
| 349c0fe4acab... |  | other | 0.00 |
| 34ce0548d06c... |  | other | 0.00 |
| 37f14b6d6d2a... |  | other | 0.00 |
| 382fb554bdb4... |  | other | 0.00 |
| 396af0dd6b50... |  | other | 0.00 |
| 3ac7926ffb27... |  | other | 0.00 |
| 3b75c8fdd9b8... |  | other | 0.00 |
| 3b9bd294ed6f... |  | other | 0.00 |
| 3bc2e1e4ad72... |  | other | 0.00 |
| 3c726426fb52... |  | other | 0.00 |
| 3db5a3338682... |  | other | 0.00 |
| 3df91d2a0d24... |  | other | 0.00 |
| 3ea274c21d79... |  | other | 0.00 |
| 3fa59589a91b... |  | other | 0.00 |
| 406365706456... |  | other | 0.00 |
| 4065b840f805... |  | other | 0.00 |
| 4153eef03837... |  | other | 0.00 |
| 417258d0c4d7... |  | other | 0.00 |
| 41da4a2ba314... |  | other | 0.00 |
| 42b2f4a738af... |  | other | 0.00 |
| 43b9033ec25f... |  | other | 0.00 |
| 44927ba58d35... |  | other | 0.00 |
| 44a42ad36d29... |  | other | 0.00 |
| 44eada8b5910... |  | other | 0.00 |
| 46d65204bc0e... |  | other | 0.00 |
| 4c433b99adaf... |  | other | 0.00 |
| 4cccd40fc2bd... |  | other | 0.00 |
| 4d4af3d59eb0... |  | other | 0.00 |
| 4d915373c993... |  | other | 0.00 |
| 4e30baac0c5e... |  | other | 0.00 |
| 4e387152960c... |  | other | 0.00 |
| 4e417e0f9e35... |  | other | 0.00 |
| 503e3d85a978... |  | other | 0.00 |
| 5065c2d1e4f9... |  | other | 0.00 |
| 51d4b9c700ef... |  | other | 0.00 |
| 520092b050db... |  | other | 0.00 |
| 52fb230e1dd6... |  | other | 0.00 |
| 536bbce854b9... |  | other | 0.00 |
| 54ba3b491049... |  | other | 0.00 |
| 55fa145c3f08... |  | other | 0.00 |
| 5b2832f111b3... |  | other | 0.00 |
| 5c3e143e80b7... |  | other | 0.00 |
| 5cf25531f59a... |  | other | 0.00 |
| 5e1dd3fe0942... |  | other | 0.00 |
| 5f2e0f8fe61a... |  | other | 0.00 |
| 5f6df5146387... |  | other | 0.00 |
| 5f90e82bc4d4... |  | other | 0.00 |
| 606129de12e0... |  | other | 0.00 |
| 60df5ac5b2e9... |  | other | 0.00 |
| 61ff6aa4d64e... |  | other | 0.00 |
| 62d4492fbd95... |  | other | 0.00 |
| 64e24ee9c86d... |  | other | 0.00 |
| 65f3f4acfcac... |  | other | 0.00 |
| 665e1be417bb... |  | other | 0.00 |
| 66923e1f9d6a... |  | other | 0.00 |
| 675e3de7c967... |  | other | 0.00 |
| 686b0d8aa480... |  | other | 0.00 |
| 686ff2967b2a... |  | other | 0.00 |
| 687b54d665d2... |  | other | 0.00 |
| 68976d0ef402... |  | other | 0.00 |
| 6aa239fca4bc... |  | other | 0.00 |
| 6c3c49e1a35a... |  | other | 0.00 |
| 6ed150324fc7... |  | other | 0.00 |
| 6f7a44b0f98c... |  | other | 0.00 |
| 70ef7d921e4a... |  | other | 0.00 |
| 717dc41cb192... |  | other | 0.00 |
| 721719c13763... |  | other | 0.00 |
| 72c68112558f... |  | other | 0.00 |
| 733522767c83... |  | other | 0.00 |
| 74d57882ef8a... |  | other | 0.00 |
| 760b864b0932... |  | other | 0.00 |
| 7762b20eeb3c... |  | other | 0.00 |
| 77746b5fbfda... |  | other | 0.00 |
| 78461539a8b7... |  | other | 0.00 |
| 7929ec9ef6ec... |  | other | 0.00 |
| 7a288e5ccc9c... |  | other | 0.00 |
| 7a3c758cb3a1... |  | other | 0.00 |
| 7a852920f26d... |  | other | 0.00 |
| 7a86c4a4082d... |  | other | 0.00 |
| 7b0f276d6be7... |  | other | 0.00 |
| 7b0f62ba9b76... |  | other | 0.00 |
| 7b30756e5285... |  | other | 0.00 |
| 7cbfd7f762ee... |  | other | 0.00 |
| 7d2514bd549b... |  | other | 0.00 |
| 7f398f457df6... |  | other | 0.00 |
| 7ff0220f6329... |  | other | 0.00 |
| 80b01206bb6c... |  | other | 0.00 |
| 8135757fe11e... |  | other | 0.00 |
| 824930648c19... |  | other | 0.00 |
| 87c3ccaba93e... |  | other | 0.00 |
| 890c80a61de8... |  | other | 0.00 |
| 891fa0be53df... |  | other | 0.00 |
| 8983927b2f64... |  | other | 0.00 |
| 8c60aac556a4... |  | other | 0.00 |
| 8c6d34f8833e... |  | other | 0.00 |
| 8e18998606b0... |  | other | 0.00 |
| 8ea39c333003... |  | other | 0.00 |
| 8f1c1c6cba05... |  | other | 0.00 |
| 8f507f768bb4... |  | other | 0.00 |
| 8f8837a8ccf2... |  | other | 0.00 |
| 91088a39fa43... |  | other | 0.00 |
| 912b9d008dbc... |  | other | 0.00 |
| 91abbc3abf55... |  | other | 0.00 |
| 9274c8c56a48... |  | other | 0.00 |
| 93880a9e97fc... |  | other | 0.00 |
| 94602bd75c48... |  | other | 0.00 |
| 96be25a94c84... |  | other | 0.00 |
| 9832c0ab41b1... |  | other | 0.00 |
| 9942e2b71471... |  | other | 0.00 |
| 99c7d4cecba3... |  | other | 0.00 |
| 9b05a9e24953... |  | other | 0.00 |
| 9b9d945ec4c6... |  | other | 0.00 |
| 9bcd21086780... |  | other | 0.00 |
| 9c522d9708f1... |  | other | 0.00 |
| 9c719be2e869... |  | other | 0.00 |
| 9e5724b81e8f... |  | other | 0.00 |
| 9f41ffdf3741... |  | other | 0.00 |
| 9f6214ef4d4b... |  | other | 0.00 |
| 9f646d76e876... |  | other | 0.00 |
| a0202a8e076c... |  | other | 0.00 |
| a09572857a46... |  | other | 0.00 |
| a3dd5a428e8b... |  | other | 0.00 |
| a47f5fc4894a... |  | other | 0.00 |
| a5c882f8a752... |  | other | 0.00 |
| a5e5dcb25404... |  | other | 0.00 |
| a62aeaeb6ecf... |  | other | 0.00 |
| a7e8a96c9239... |  | other | 0.00 |
| a9c428eef902... |  | other | 0.00 |
| ab85799d22c0... |  | other | 0.00 |
| ab9dcd09d0bc... |  | other | 0.00 |
| ace28d887202... |  | other | 0.00 |
| ad76dc709385... |  | other | 0.00 |
| ae27f8091dab... |  | other | 0.00 |
| af041f578afc... |  | other | 0.00 |
| b01c4920a4aa... |  | other | 0.00 |
| b02055aab210... |  | other | 0.00 |
| b06ff01eb2cb... |  | other | 0.00 |
| b212f401e34d... |  | other | 0.00 |
| b395a50e4a7a... |  | other | 0.00 |
| b486db089615... |  | other | 0.00 |
| b4d7a422e0ed... |  | other | 0.00 |
| b6490fdcc13f... |  | other | 0.00 |
| b65b18d52daf... |  | other | 0.00 |
| b6d097952fea... |  | other | 0.00 |
| b9cde965ff21... |  | other | 0.00 |
| b9fdb42c55d6... |  | other | 0.00 |
| bad9ca69916e... |  | other | 0.00 |
| bcb45cace93d... |  | other | 0.00 |
| bcbf1b4067fd... |  | other | 0.00 |
| bd2d222ee0a6... |  | other | 0.00 |
| bd427f328c98... |  | other | 0.00 |
| be66479a8d8e... |  | other | 0.00 |
| bf128ffae822... |  | other | 0.00 |
| c01f4b2bf7e3... |  | other | 0.00 |
| c19335caa756... |  | other | 0.00 |
| c19bcb15e6c9... |  | other | 0.00 |
| c3526451a913... |  | other | 0.00 |
| c423ebbf17c6... |  | other | 0.00 |
| ca0ca98faf97... |  | other | 0.00 |
| cbaf234440cc... |  | other | 0.00 |
| ce2f02347ea9... |  | other | 0.00 |
| cf18d18dae35... |  | other | 0.00 |
| cf595de1ee13... |  | other | 0.00 |
| cf6fe826c8d9... |  | other | 0.00 |
| cffda548fa3f... |  | other | 0.00 |
| d32e76d495ac... |  | other | 0.00 |
| d3d8538e21d8... |  | other | 0.00 |
| d403315f013e... |  | other | 0.00 |
| d55274d6a16d... |  | other | 0.00 |
| d592fb6165ea... |  | other | 0.00 |
| d840457fd538... |  | other | 0.00 |
| d8a621706e30... |  | other | 0.00 |
| d9323cf2c71d... |  | other | 0.00 |
| dbe11c5a8ded... |  | other | 0.00 |
| dc1859b7e895... |  | other | 0.00 |
| dc942610b9f6... |  | other | 0.00 |
| dcdfdb7338f8... |  | other | 0.00 |
| dda8c7ff4dc7... |  | other | 0.00 |
| ddd12664d040... |  | other | 0.00 |
| df1355930bf2... |  | other | 0.00 |
| e2e3deaf1cce... |  | other | 0.00 |
| e431876e23e8... |  | other | 0.00 |
| e431a7be5d22... |  | other | 0.00 |
| e46e9d9e140e... |  | other | 0.00 |
| e4741149b9a4... |  | other | 0.00 |
| e4cd798b4b91... |  | other | 0.00 |
| e5141e7c3c20... |  | other | 0.00 |
| e62abd09e7a4... |  | other | 0.00 |
| e70ae1d37abe... |  | other | 0.00 |
| e8f40f6102b8... |  | other | 0.00 |
| e9a32d22eb9b... |  | other | 0.00 |
| ea81bf0f8c1d... |  | other | 0.00 |
| eaa0fe6459ac... |  | other | 0.00 |
| eb035930bcd0... |  | other | 0.00 |
| eb21473f54ab... |  | other | 0.00 |
| ebb6ef8d0280... |  | other | 0.00 |
| ec70dca8ed05... |  | other | 0.00 |
| ecb090f111a0... |  | other | 0.00 |
| ed04d28060b0... |  | other | 0.00 |
| ee0796ad2bae... |  | other | 0.00 |
| efa6f9ce52c6... |  | other | 0.00 |
| f0681f1c30de... |  | other | 0.00 |
| f11e77a89f31... |  | other | 0.00 |
| f283b88843e9... |  | other | 0.00 |
| f2a738454f11... |  | other | 0.00 |
| f3ef6ad7f523... |  | other | 0.00 |
| f496d28b9b6d... |  | other | 0.00 |
| f4ade2ce28f5... |  | other | 0.00 |
| f4c4d051d514... |  | other | 0.00 |
| f8d3778ec289... |  | other | 0.00 |
| f954e46dee6f... |  | other | 0.00 |
| fa251d4bbd68... |  | other | 0.00 |
| fb2d4e5fc406... |  | other | 0.00 |
| fbcf7b546efd... |  | other | 0.00 |
| fbdd7f9dd299... |  | other | 0.00 |
| fccbe5d824f0... |  | other | 0.00 |
| fe8332161cbb... |  | other | 0.00 |
| ff50fa9d2c1a... |  | other | 0.00 |
| ff746f7b5834... |  | other | 0.00 |
| ff79b04bb443... |  | other | 0.00 |

## Document Type Comparison

| Doc Type | Private | Warmup | Delta |
|----------|---------|--------|-------|
| amendment_law | 2 | 2 | 0 |
| case_law | 16 | 0 | +16 |
| court_case | 0 | 16 | -16 |
| enactment_notice | 5 | 5 | 0 |
| law | 16 | 16 | 0 |
| other | 279 | 0 | +279 |

## Predicted Failure Hotspots

- **New document types**: case_law, other -- no ingestion heuristics tuned
- **High-complexity questions**: 1 questions with complexity >= 5 (multi-hop, multi-entity)
- **Low warmup similarity**: 59 questions with token Jaccard < 0.15 vs any warmup question
- **Broad free-text without anchors**: 225 questions -- retrieval will rely solely on embedding similarity
