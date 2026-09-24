# Error Analysis

## Per-Class Error Rates

| model | govt_critique | govt_leaning | neutral |
| --- | --- | --- | --- |
| align | 0.5385 | 0.5000 | 0.4828 |
| bangla_electra | 0.8641 | 0.2857 | 0.4340 |
| banglabert | 0.2913 | 0.5238 | 0.6038 |
| banglabert_frozen | 0.4757 | 0.5476 | 0.6981 |
| banglabert_no_aug | 0.2913 | 0.5238 | 0.6038 |
| blip | 0.3846 | 0.8889 | 0.6207 |
| blip_unfrozen | 0.6154 | 0.6667 | 0.4138 |
| clip | 0.0513 | 0.8333 | 0.5517 |
| clip_unfrozen | 0.2564 | 0.7778 | 0.3448 |
| countvec_vit | 0.4359 | 0.7778 | 0.5172 |
| flava | 0.4103 | 1.0000 | 0.6207 |
| majority | 0.0000 | 1.0000 | 1.0000 |
| mbert | 0.4272 | 0.4048 | 0.6415 |
| mt5 | 0.2816 | 0.5476 | 0.9434 |
| tfidf_logreg | 0.3398 | 0.6667 | 0.4906 |
| tfidf_logreg_no_aug | 0.3398 | 0.6667 | 0.4906 |
| vilt | 0.4872 | 0.7778 | 0.7241 |
| xlmr | 0.6699 | 0.2857 | 0.6226 |


## Hardest Items (most models wrong)

- **Image_32** (100% wrong, 10/10 models) true=neutral, outlet=Jugantor
  - Headline: দাম বাড়বে ইট-সিমেন্টের
  - Predictions: bangla_electra=govt_critique; banglabert=govt_critique; banglabert_frozen=govt_critique; banglabert_no_aug=govt_critique; majority=govt_critique; mbert=govt_critique; mt5=govt_critique; tfidf_logreg=g...
- **Image_39** (100% wrong, 10/10 models) true=neutral, outlet=The Daily Star
  - Headline: ১০ ডিসেম্বর: একটি রাজনৈতিক সমাবেশ নিয়ে কেন এত তর্ক-বিতর্ক
  - Predictions: bangla_electra=govt_critique; banglabert=govt_critique; banglabert_frozen=govt_critique; banglabert_no_aug=govt_critique; majority=govt_critique; mbert=govt_critique; mt5=govt_critique; tfidf_logreg=g...
- **Image_77** (100% wrong, 10/10 models) true=neutral, outlet=The Daily Star
  - Headline: সোমবার নয়, বৃহস্পতিবার থেকে সারাদেশে কঠোর লকডাউন
  - Predictions: bangla_electra=govt_leaning; banglabert=govt_critique; banglabert_frozen=govt_critique; banglabert_no_aug=govt_critique; majority=govt_critique; mbert=govt_critique; mt5=govt_critique; tfidf_logreg=go...
- **Image_182** (100% wrong, 10/10 models) true=neutral, outlet=Samakal
  - Headline: মিথ্যা আশ্বাস নয়, বাস্তব পরিকল্পনা দেবে জামায়াত: শফিকুর রহমান
  - Predictions: bangla_electra=govt_leaning; banglabert=govt_critique; banglabert_frozen=govt_critique; banglabert_no_aug=govt_critique; majority=govt_critique; mbert=govt_leaning; mt5=govt_leaning; tfidf_logreg=govt...
- **Image_95** (90% wrong, 9/10 models) true=neutral, outlet=Bangla News24
  - Headline: ডিজিটাল সিকিউরিটি অ্যাক্ট সংশোধনের ইঙ্গিত আইনমন্ত্রীর
  - Predictions: bangla_electra=govt_leaning; banglabert=govt_critique; banglabert_frozen=neutral; banglabert_no_aug=govt_critique; majority=govt_critique; mbert=govt_critique; mt5=govt_leaning; tfidf_logreg=govt_crit...
- **Image_200** (90% wrong, 9/10 models) true=neutral, outlet=prothom Alo
  - Headline: পাকিস্তানে শতাধিক শিশুকে যৌন নিপীড়নের অভিযোগে করাচিতে গ্রেপ্তার
  - Predictions: bangla_electra=govt_leaning; banglabert=govt_critique; banglabert_frozen=govt_critique; banglabert_no_aug=govt_critique; majority=govt_critique; mbert=neutral; mt5=govt_critique; tfidf_logreg=govt_cri...
- **Image_54** (90% wrong, 9/10 models) true=govt_leaning, outlet=Somoy news
  - Headline: ডিজেলের দাম বৃদ্ধি, যে ব্যাখ্যা দিল মন্ত্রণালয়
  - Predictions: bangla_electra=neutral; banglabert=govt_critique; banglabert_frozen=govt_critique; banglabert_no_aug=govt_critique; majority=govt_critique; mbert=govt_leaning; mt5=govt_critique; tfidf_logreg=neutral;...
- **Image_184** (90% wrong, 9/10 models) true=neutral, outlet=prothom Alo
  - Headline: স্বাস্থ্য বিভাগের অভিযান
কারও লাইসেন্স নেই, কারও ফ্রিজে ছিল মাছ-মাংস, ১১টি ক্লিন
  - Predictions: bangla_electra=neutral; banglabert=govt_critique; banglabert_frozen=govt_critique; banglabert_no_aug=govt_critique; majority=govt_critique; mbert=govt_leaning; mt5=govt_critique; tfidf_logreg=govt_cri...
- **Image_24** (90% wrong, 9/10 models) true=neutral, outlet=Cvoice24
  - Headline: ৪ ধারা অজামিনযোগ্য রেখে সাইবার নিরাপত্তা বিল
  - Predictions: bangla_electra=neutral; banglabert=govt_leaning; banglabert_frozen=govt_leaning; banglabert_no_aug=govt_leaning; majority=govt_critique; mbert=govt_leaning; mt5=govt_leaning; tfidf_logreg=govt_critiqu...
- **Image_59** (90% wrong, 9/10 models) true=govt_leaning, outlet=Jugantor
  - Headline: পদ্মা সেতুর কাজ মাত্র ৫ শতাংশ বাকি
  - Predictions: bangla_electra=govt_leaning; banglabert=govt_critique; banglabert_frozen=govt_critique; banglabert_no_aug=govt_critique; majority=govt_critique; mbert=neutral; mt5=govt_critique; tfidf_logreg=neutral;...
- **Image_105** (89% wrong, 16/18 models) true=govt_leaning, outlet=Channel online
  - Headline: কোটা সংস্কার আন্দোলন, টার্গেট কারা?
  - Predictions: align=govt_critique; bangla_electra=govt_leaning; banglabert=govt_critique; banglabert_frozen=govt_critique; banglabert_no_aug=govt_critique; blip=govt_critique; blip_unfrozen=govt_critique; clip=govt...
- **Image_149** (89% wrong, 16/18 models) true=govt_leaning, outlet=BBC Bangla
  - Headline: ক্যাসিনোতে র‍্যাবের অভিযান: জুয়া খেলা নিয়ে আইনে যা আছে
  - Predictions: align=govt_critique; bangla_electra=govt_leaning; banglabert=neutral; banglabert_frozen=govt_critique; banglabert_no_aug=neutral; blip=govt_critique; blip_unfrozen=govt_critique; clip=govt_critique; c...
- **Image_114** (83% wrong, 15/18 models) true=govt_leaning, outlet=BBC Bangla
  - Headline: রোহিঙ্গা' ও 'শরণার্থী' বলবে না বাংলাদেশ সরকার
  - Predictions: align=govt_leaning; bangla_electra=govt_critique; banglabert=govt_critique; banglabert_frozen=neutral; banglabert_no_aug=govt_critique; blip=govt_critique; blip_unfrozen=neutral; clip=neutral; clip_un...
- **Image_151** (83% wrong, 15/18 models) true=govt_leaning, outlet=dainik ittefaq
  - Headline: সময়ের আগেই শেষ হলো পদ্মা রেল প্রকল্প; ১,৮৪৫ কোটি টাকা সাশ্রয়
  - Predictions: align=govt_leaning; bangla_electra=govt_leaning; banglabert=govt_critique; banglabert_frozen=govt_critique; banglabert_no_aug=govt_critique; blip=govt_critique; blip_unfrozen=govt_critique; clip=govt_...
- **Image_18** (83% wrong, 15/18 models) true=govt_leaning, outlet=Daily Ittefaq
  - Headline: সার্বজনীন পেনশন স্কিম নিয়ে সরকারের আনুষ্ঠানিক বক্তব্য
  - Predictions: align=neutral; bangla_electra=govt_leaning; banglabert=govt_critique; banglabert_frozen=neutral; banglabert_no_aug=govt_critique; blip=neutral; blip_unfrozen=govt_leaning; clip=govt_critique; clip_unf...
- **Image_165** (83% wrong, 15/18 models) true=neutral, outlet=Prothom Alo
  - Headline: প্রধানমন্ত্রীকে চিঠি
ডিজিটাল নিরাপত্তা আইনের সব মামলা বাতিলের দাবি ১৯ আন্তর্জাতি
  - Predictions: align=govt_leaning; bangla_electra=neutral; banglabert=govt_critique; banglabert_frozen=neutral; banglabert_no_aug=govt_critique; blip=govt_critique; blip_unfrozen=govt_critique; clip=neutral; clip_un...
- **Image_140** (83% wrong, 15/18 models) true=neutral, outlet=BBC Bangla
  - Headline: ফিরে দেখা: হেফাজতে ইসলামের ঢাকা অবরোধ
  - Predictions: align=govt_critique; bangla_electra=neutral; banglabert=govt_critique; banglabert_frozen=govt_critique; banglabert_no_aug=govt_critique; blip=neutral; blip_unfrozen=neutral; clip=govt_critique; clip_u...
- **Image_134** (83% wrong, 15/18 models) true=govt_leaning, outlet=DW
  - Headline: বিনা প্রতিদ্বন্দ্বিতায় সাংসদ নির্বাচন
  - Predictions: align=neutral; bangla_electra=govt_leaning; banglabert=neutral; banglabert_frozen=neutral; banglabert_no_aug=neutral; blip=neutral; blip_unfrozen=govt_critique; clip=govt_critique; clip_unfrozen=govt_...
- **Image_193** (80% wrong, 8/10 models) true=neutral, outlet=BBC Bangla
  - Headline: ২০০৭ সালের জুলাইয়ে যেভাবে গ্রেফতার হয়েছিলেন শেখ হাসিনা
  - Predictions: bangla_electra=neutral; banglabert=govt_critique; banglabert_frozen=govt_leaning; banglabert_no_aug=govt_critique; majority=govt_critique; mbert=govt_critique; mt5=govt_critique; tfidf_logreg=govt_lea...
- **Image_43** (80% wrong, 8/10 models) true=neutral, outlet=Bangla Tribune
  - Headline: র‌্যাবের ২৩৬ সদস্যের বিরুদ্ধে শাস্তিমূলক ব্যবস্থা
  - Predictions: bangla_electra=govt_leaning; banglabert=govt_critique; banglabert_frozen=govt_critique; banglabert_no_aug=govt_critique; majority=govt_critique; mbert=govt_critique; mt5=govt_critique; tfidf_logreg=ne...


## Accuracy by Outlet

| outlet | mean_acc | n |
| --- | --- | --- |
| Chalaman neywork | 0.8000 | 1 |
| Dainik amader bangla | 0.8000 | 1 |
| voabangla | 0.8000 | 1 |
| the bangladesh today | 0.7778 | 1 |
| Dhaka Tribune | 0.6481 | 2 |
| Daily Inqilab | 0.6389 | 1 |
| The Daily Star | 0.6181 | 1 |
| The doctors dialogue | 0.6111 | 1 |
| Kaler kontho | 0.6111 | 1 |
| Ajkaler khobor | 0.6111 | 1 |
| smsaif | 0.6111 | 1 |
| Desh Rupantor | 0.6000 | 1 |
| Bangla News24 | 0.5972 | 1 |
| ekhon tv | 0.5556 | 1 |
| Somoyer Alo | 0.5556 | 1 |
| Dhaka Post | 0.5370 | 1 |
| BBC Bangla | 0.5105 | 15 |
| Bdnews24 | 0.5093 | 4 |
| Prothom Alo | 0.5000 | 23 |
| Khulna Gazet | 0.5000 | 1 |
| Daily Janakantho | 0.5000 | 1 |
| rtv online | 0.5000 | 1 |
| BBC bangla | 0.5000 | 1 |
| Bangla Tribune | 0.4944 | 6 |
| Samakal | 0.4926 | 3 |
| DW | 0.4917 | 5 |
| Somoy news | 0.4907 | 2 |
| dainik ittefaq | 0.4722 | 2 |
| Doinik Bangla | 0.4444 | 1 |
| DBC news | 0.4444 | 1 |
| Jago News | 0.4200 | 5 |
| Dhaka Times | 0.4167 | 1 |
| Daily Ittefaq | 0.3981 | 2 |
| Banik Barta | 0.3889 | 1 |
| Channel online | 0.3333 | 2 |
| Jugantor | 0.3276 | 3 |
| prothom Alo | 0.3250 | 4 |
| Bdnew24 | 0.3000 | 1 |
| News Bangla | 0.3000 | 1 |
| Kaler Kontho | 0.3000 | 1 |
| the daily campus | 0.2778 | 1 |
| Cvoice24 | 0.1000 | 1 |


## Accuracy by Headline Length

| len_bin | mean_acc | total_items |
| --- | --- | --- |
| 100-200 | 0.2716 | 98 |
| 30-60 | 0.4885 | 1506 |
| 60-100 | 0.5070 | 914 |
| <30 | 0.4737 | 150 |

