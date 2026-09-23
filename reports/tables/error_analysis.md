# Error Analysis

## Per-Class Error Rates

| model | govt_critique | govt_leaning | neutral |
| --- | --- | --- | --- |
| align | 0.5897 | 0.6111 | 0.5517 |
| bangla_electra | 0.8641 | 0.2857 | 0.4340 |
| banglabert | 0.2913 | 0.5238 | 0.6038 |
| blip | 0.3846 | 0.8889 | 0.6207 |
| clip | 0.0513 | 0.8333 | 0.5517 |
| countvec_vit | 0.4359 | 0.7778 | 0.5172 |
| flava | 0.4103 | 1.0000 | 0.6207 |
| majority | 0.0000 | 1.0000 | 1.0000 |
| mbert | 0.4272 | 0.4048 | 0.6415 |
| mt5 | 0.4466 | 0.3571 | 0.6415 |
| tfidf_logreg | 0.3398 | 0.6667 | 0.4906 |
| vilt | 0.4872 | 0.7778 | 0.7241 |
| xlmr | 0.6699 | 0.2857 | 0.6226 |


## Hardest Items (most models wrong)

- **Image_95** (100% wrong, 7/7 models) true=neutral, outlet=Bangla News24
  - Headline: ডিজিটাল সিকিউরিটি অ্যাক্ট সংশোধনের ইঙ্গিত আইনমন্ত্রীর
  - Predictions: bangla_electra=govt_leaning; banglabert=govt_critique; majority=govt_critique; mbert=govt_critique; mt5=govt_leaning; tfidf_logreg=govt_critique; xlmr=govt_leaning
- **Image_32** (100% wrong, 7/7 models) true=neutral, outlet=Jugantor
  - Headline: দাম বাড়বে ইট-সিমেন্টের
  - Predictions: bangla_electra=govt_critique; banglabert=govt_critique; majority=govt_critique; mbert=govt_critique; mt5=govt_leaning; tfidf_logreg=govt_critique; xlmr=govt_leaning
- **Image_43** (86% wrong, 6/7 models) true=neutral, outlet=Bangla Tribune
  - Headline: র‌্যাবের ২৩৬ সদস্যের বিরুদ্ধে শাস্তিমূলক ব্যবস্থা
  - Predictions: bangla_electra=govt_leaning; banglabert=govt_critique; majority=govt_critique; mbert=govt_critique; mt5=govt_critique; tfidf_logreg=neutral; xlmr=govt_critique
- **Image_91** (86% wrong, 6/7 models) true=govt_leaning, outlet=Kaler Kontho
  - Headline: ক্যাসিনো ব্যবসা, শুদ্ধি অভিযান প্রতিক্রিয়া ও প্রত্যাশা
  - Predictions: bangla_electra=neutral; banglabert=govt_critique; majority=govt_critique; mbert=govt_critique; mt5=neutral; tfidf_logreg=govt_leaning; xlmr=neutral
- **Image_47** (86% wrong, 6/7 models) true=neutral, outlet=BBC Bangla
  - Headline: ডিসেম্বরে চালু হতে পারে রামপাল বিদ্যুৎকেন্দ্র, পরিবেশ বিপর্যয়ের ঝুঁকি থাকছেই
  - Predictions: bangla_electra=govt_leaning; banglabert=neutral; majority=govt_critique; mbert=govt_leaning; mt5=govt_critique; tfidf_logreg=govt_critique; xlmr=govt_leaning
- **Image_77** (86% wrong, 6/7 models) true=neutral, outlet=The Daily Star
  - Headline: সোমবার নয়, বৃহস্পতিবার থেকে সারাদেশে কঠোর লকডাউন
  - Predictions: bangla_electra=govt_leaning; banglabert=govt_critique; majority=govt_critique; mbert=govt_critique; mt5=neutral; tfidf_logreg=govt_critique; xlmr=govt_leaning
- **Image_184** (86% wrong, 6/7 models) true=neutral, outlet=prothom Alo
  - Headline: স্বাস্থ্য বিভাগের অভিযান
কারও লাইসেন্স নেই, কারও ফ্রিজে ছিল মাছ-মাংস, ১১টি ক্লিন
  - Predictions: bangla_electra=neutral; banglabert=govt_critique; majority=govt_critique; mbert=govt_leaning; mt5=govt_critique; tfidf_logreg=govt_critique; xlmr=govt_leaning
- **Image_39** (86% wrong, 6/7 models) true=neutral, outlet=The Daily Star
  - Headline: ১০ ডিসেম্বর: একটি রাজনৈতিক সমাবেশ নিয়ে কেন এত তর্ক-বিতর্ক
  - Predictions: bangla_electra=govt_critique; banglabert=govt_critique; majority=govt_critique; mbert=govt_critique; mt5=neutral; tfidf_logreg=govt_critique; xlmr=govt_critique
- **Image_200** (86% wrong, 6/7 models) true=neutral, outlet=prothom Alo
  - Headline: পাকিস্তানে শতাধিক শিশুকে যৌন নিপীড়নের অভিযোগে করাচিতে গ্রেপ্তার
  - Predictions: bangla_electra=govt_leaning; banglabert=govt_critique; majority=govt_critique; mbert=neutral; mt5=govt_critique; tfidf_logreg=govt_critique; xlmr=govt_critique
- **Image_53** (86% wrong, 6/7 models) true=govt_critique, outlet=Prothom Alo
  - Headline: বাড়ল জ্বালানির দাম দুঃসময়ে জীবনযাত্রায় আরও চাপ
  - Predictions: bangla_electra=govt_leaning; banglabert=neutral; majority=govt_critique; mbert=govt_leaning; mt5=neutral; tfidf_logreg=neutral; xlmr=neutral
- **Image_37** (86% wrong, 6/7 models) true=neutral, outlet=Samakal
  - Headline: নেতৃত্বেই গোলমেলে রাজশাহী ছাত্রদল
  - Predictions: bangla_electra=neutral; banglabert=govt_critique; majority=govt_critique; mbert=govt_leaning; mt5=govt_critique; tfidf_logreg=govt_critique; xlmr=govt_leaning
- **Image_24** (86% wrong, 6/7 models) true=neutral, outlet=Cvoice24
  - Headline: ৪ ধারা অজামিনযোগ্য রেখে সাইবার নিরাপত্তা বিল
  - Predictions: bangla_electra=neutral; banglabert=govt_leaning; majority=govt_critique; mbert=govt_leaning; mt5=govt_leaning; tfidf_logreg=govt_critique; xlmr=govt_leaning
- **Image_59** (86% wrong, 6/7 models) true=govt_leaning, outlet=Jugantor
  - Headline: পদ্মা সেতুর কাজ মাত্র ৫ শতাংশ বাকি
  - Predictions: bangla_electra=govt_leaning; banglabert=govt_critique; majority=govt_critique; mbert=neutral; mt5=neutral; tfidf_logreg=neutral; xlmr=neutral
- **Image_182** (86% wrong, 6/7 models) true=neutral, outlet=Samakal
  - Headline: মিথ্যা আশ্বাস নয়, বাস্তব পরিকল্পনা দেবে জামায়াত: শফিকুর রহমান
  - Predictions: bangla_electra=govt_leaning; banglabert=govt_critique; majority=govt_critique; mbert=govt_leaning; mt5=neutral; tfidf_logreg=govt_leaning; xlmr=govt_critique
- **Image_114** (85% wrong, 11/13 models) true=govt_leaning, outlet=BBC Bangla
  - Headline: রোহিঙ্গা' ও 'শরণার্থী' বলবে না বাংলাদেশ সরকার
  - Predictions: align=govt_critique; bangla_electra=govt_critique; banglabert=govt_critique; blip=govt_critique; clip=neutral; countvec_vit=neutral; flava=govt_critique; majority=govt_critique; mbert=govt_leaning; mt...
- **Image_140** (85% wrong, 11/13 models) true=neutral, outlet=BBC Bangla
  - Headline: ফিরে দেখা: হেফাজতে ইসলামের ঢাকা অবরোধ
  - Predictions: align=govt_leaning; bangla_electra=neutral; banglabert=govt_critique; blip=neutral; clip=govt_critique; countvec_vit=govt_critique; flava=govt_critique; majority=govt_critique; mbert=govt_leaning; mt5...
- **Image_165** (85% wrong, 11/13 models) true=neutral, outlet=Prothom Alo
  - Headline: প্রধানমন্ত্রীকে চিঠি
ডিজিটাল নিরাপত্তা আইনের সব মামলা বাতিলের দাবি ১৯ আন্তর্জাতি
  - Predictions: align=govt_leaning; bangla_electra=neutral; banglabert=govt_critique; blip=govt_critique; clip=neutral; countvec_vit=govt_critique; flava=govt_critique; majority=govt_critique; mbert=govt_critique; mt...
- **Image_105** (85% wrong, 11/13 models) true=govt_leaning, outlet=Channel online
  - Headline: কোটা সংস্কার আন্দোলন, টার্গেট কারা?
  - Predictions: align=govt_leaning; bangla_electra=govt_leaning; banglabert=govt_critique; blip=govt_critique; clip=govt_critique; countvec_vit=govt_critique; flava=govt_critique; majority=govt_critique; mbert=govt_c...
- **Image_142** (85% wrong, 11/13 models) true=govt_leaning, outlet=Bdnews24
  - Headline: হরতাল প্রতিরোধে জাগরণের মিছিল
  - Predictions: align=govt_critique; bangla_electra=neutral; banglabert=govt_leaning; blip=govt_critique; clip=govt_critique; countvec_vit=govt_critique; flava=neutral; majority=govt_critique; mbert=govt_critique; mt...
- **Image_143** (77% wrong, 10/13 models) true=govt_leaning, outlet=Jugantor
  - Headline: হেফাজত নেতাদের নজরদারির পরামর্শ একদফা আন্দোলনে নামানোর পরিকল্পনা চলছে * হাটহাজার
  - Predictions: align=neutral; bangla_electra=neutral; banglabert=govt_leaning; blip=govt_critique; clip=govt_critique; countvec_vit=neutral; flava=govt_critique; majority=govt_critique; mbert=govt_leaning; mt5=govt_...


## Accuracy by Outlet

| outlet | mean_acc | n |
| --- | --- | --- |
| the bangladesh today | 0.7692 | 1 |
| Chalaman neywork | 0.7143 | 1 |
| Desh Rupantor | 0.7143 | 1 |
| voabangla | 0.7143 | 1 |
| Dainik amader bangla | 0.7143 | 1 |
| The Daily Star | 0.6827 | 1 |
| Bangla News24 | 0.6346 | 1 |
| Ajkaler khobor | 0.6154 | 1 |
| smsaif | 0.6154 | 1 |
| Daily Inqilab | 0.6154 | 1 |
| Dhaka Tribune | 0.6154 | 2 |
| Daily Janakantho | 0.5714 | 1 |
| rtv online | 0.5385 | 1 |
| Kaler kontho | 0.5385 | 1 |
| Somoy news | 0.5256 | 2 |
| Prothom Alo | 0.4989 | 23 |
| Jago News | 0.4857 | 5 |
| DW | 0.4846 | 5 |
| BBC Bangla | 0.4791 | 15 |
| Dhaka Post | 0.4744 | 1 |
| dainik ittefaq | 0.4615 | 2 |
| ekhon tv | 0.4615 | 1 |
| Doinik Bangla | 0.4615 | 1 |
| The doctors dialogue | 0.4615 | 1 |
| Bangla Tribune | 0.4590 | 6 |
| Bdnews24 | 0.4551 | 4 |
| Samakal | 0.4410 | 3 |
| Khulna Gazet | 0.4286 | 1 |
| Somoyer Alo | 0.4231 | 1 |
| Daily Ittefaq | 0.4103 | 2 |
| DBC news | 0.3846 | 1 |
| Dhaka Times | 0.3846 | 1 |
| Banik Barta | 0.3846 | 1 |
| Jugantor | 0.3590 | 3 |
| Channel online | 0.3462 | 2 |
| prothom Alo | 0.3214 | 4 |
| the daily campus | 0.3077 | 1 |
| BBC bangla | 0.2857 | 1 |
| Bdnew24 | 0.2857 | 1 |
| News Bangla | 0.2857 | 1 |
| Kaler Kontho | 0.1429 | 1 |
| Cvoice24 | 0.1429 | 1 |


## Accuracy by Headline Length

| len_bin | mean_acc | total_items |
| --- | --- | --- |
| 100-200 | 0.2650 | 69 |
| 30-60 | 0.4795 | 1073 |
| 60-100 | 0.4891 | 653 |
| <30 | 0.4056 | 107 |

