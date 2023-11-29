from transformers import AutoTokenizer
from exllamav2 import ExLlamaV2Config
from exllamav2 import ExLlamaV2Tokenizer
import random

model_path = "/mnt/str/models/_exl2/deepseek-coder-1.3b"

config = ExLlamaV2Config()
config.model_dir = model_path
config.prepare()
exl2_tokenizer = ExLlamaV2Tokenizer(config)

reference_tokenizer = AutoTokenizer.from_pretrained(model_path)

# Bunch of text

text = """

Following added tokens are encoded correctly but decoded incorrectly by the HF tokenizer:

ö
÷
Á
ý
À
ÿ
ø
ú
þ
ü
ù
ö
û

(workaround seems to be working for now)



This command might take a bit of time if your corpus is very large, but for this dataset of 1.6 GB of texts it’s blazing fast (1 minute 16 seconds on an AMD Ryzen 9
 3900X CPU with 12 cores). Note that AutoTokenizer.train_new_from_iterator() only works if the tokenizer you are using is a “fast” tokenizer. As you’ll see in the next section,
  the 🤗 Transformers library contains two types of tokenizers: some are written purely in Python and others (the fast ones) are backed by the 🤗 Tokenizers library, which is
   written in the Rust programming language. Python is the language most often used for data science and deep learning applications, but when anything needs to be parallelized
    to be fast, it has to be written in another language. For instance, the matrix multiplications that are at the core of the model computation are written in CUDA, an 
   optimized C library for GPUs.
Training a brand new tokenizer in pure Python would be excruciating
a language model is not available in the language you are interested in, or if your corpus is very different from the one your language model was trained on, you will most 
likely want to retrain the model from scratch using a tokenizer adapted to your data. That will require training a new tokenizer on your dataset. But what exactly does that
 mean? When we first looked at tokenizers in Chapter 2, we saw that most Transformer models use a subword tokenization algorithm. To identify which subwords are of interest
  and occur most frequently in the corpus at hand, the tokenizer needs to take a hard look at all the texts in the corpus — a process we call training. The exact rules that
   govern thi


!
!asdasd
!asdfasdas asd asd asd asd 
xx                                                                                              xxxx
#######################################################################################################
*✰
⋚⊹
(╬⁽⁽ ⁰ ⁾⁾ Д ⁽⁽ ⁰ ⁾⁾)
(●´⌓`●)
(シ_ _)シ
m(_ _)m
(＾-＾)＿日
((유∀유|||))
(๑´• .̫ •ू`๑)
.°(ಗдಗ。)°.
(Θ︹Θ)ს
◝(๑꒪່౪̮꒪່๑)◜
(ᗒᗣᗕ)՞
ू(ʚ̴̶̷́ .̠ ʚ̴̶̷̥̀ ू)
(┳Д┳)
(ᵒ̤̑ ₀̑ ᵒ̤̑)wow
٩(⌯꒦ິ̆ᵔ꒦ິ)۶ᵒᵐᵍᵎᵎᵎ
⊙▂⊙
(´⊙ω
(◐ω◑ )
ლ(́◉◞౪◟◉‵ლ)
(´°̥̥̥̥̥̥̥̥ω°̥̥̥̥̥̥̥̥｀)
꒰ღ˘‿˘ற꒱❤⃛
-(๑☆‿ ☆#)ᕗ
⸂⸂⸜(രᴗര๑)⸝⸃⸃
٩(•̤̀ᵕ•̤́๑)ᵒᵏᵎᵎᵎᵎ
(✌ﾟ∀ﾟ)☞
(ง'̀-'́)ง
( ˘▽˘)っ♨

ly slow, which is why we developed the 🤗 Tokenizers library. Note that just as you didn’t have to learn the CUDA language to be able to execute your model on a batch of i
nputs on a GPU, you won’t need to learn Rust to use a fast tokenizer. The 🤗 Tokenizers library provides Python bindings for many methods that internally call some piece o
f code in Rust; for example, to parallelize the training of your new tokenizer or, as we saw in Chapter 3, the tokenization of a batch of inputs.
Most of the Transformer models have a fast tokenizer available (there are some exceptions that you can check here), and the AutoTokenizer API always selects the 
fast tokenizer for you if it’s available. In the next section we’ll take a look at some of the other special features fast tokenizers have, which will be really useful f
or tasks like token classification and question answering. Before diving into that, however, let’s try our brand new tokenizer on the previous example:
tokens = tokenizer.tokenize(example)
tokens
(˵¯͒〰¯͒˵)
೭੧(❛〜❛✿)੭೨
(,,◕　⋏　◕,,)
( ؕؔʘ̥̥̥̥ ه ؔؕʘ̥̥̥̥ )?
ಠ_ರೃ
♥╣[-_-]╠♥
(∿°○°)∿ ︵ ǝʌol
（。ˇ ⊖ˇ）♡
‧⁺◟( ᵒ̴̶̷̥́ ·̫ ᵒ̴̶̷̣̥̀ )
꒰⌗´͈ ᵕ ॣ`͈⌗꒱৩
(ᵒ̴̶̷̤́◞౪◟ ᵒ̴̶̷̤̀ )

['def', 'Ġadd', '_', 'numbers', '(', 'a', ',', 'Ġb', '):', 'ĊĠĠĠ', 'Ġ, 'Add', 'Ġthe', 'Ġtwo', 'Ġnumbers', 'Ġ`',
 'a', '`', 'Ġand', 'Ġ`', 'b', '`.'ĊĠĠĠ', 'Ġreturn', 'Ġa', 'Ġ+', 'Ġb']
Here we again see the special symbols Ġ and Ċ that denote spaces and newlines, but we can also see that our tokenizer learned some tokens that are highly specific to 
a corpus of
使用当今最常用的分词器来训练新词汇并进行分词。
     由于 Rust 实现，速度非常快（训练和标记化）。 在服务器 CPU 上标记 1 GB 文本只需不到 20 秒。
     易于使用，而且用途极其广泛。
     专为研究和生产而设计。
     标准化伴随着对齐跟踪。 总是可以获得原始句子中与给定标记相对应的部分。
     执行所有预处理：截断、填充、添加模型所需的特殊标记。
使用當今最常用的分詞器來訓練新詞彙並進行分詞。
     由於 Rust 實現，速度非常快（訓練和標記化）。 在伺服器 CPU 上標記 1 GB 文字只需不到 20 秒。
     易於使用，而且用途極為廣泛。
     專為研究和生產而設計。
     標準化伴隨著對齊追蹤。 總是可以獲得原始句子中與給定標記相對應的部分。
     執行所有預處理：截斷、填充、新增模型所需的特殊標記。
     Išmokykite naujus žodynus ir žetonus naudodami šiandien dažniausiai naudojamus žetonus.
     Itin greitas (ir mokymas, ir tokenizavimas), dėka Rust įgyvendinimo. GB teksto atpažinimas serverio procesoriuje trunka mažiau nei 20 sekundžių.
     Lengva naudoti, bet ir itin universalus.
     Sukurta tyrimams ir gamybai.
     Normalizacija ateina su derinimo stebėjimu. Visada galima gauti pradinio sakinio dalį, atitinkančią duotą žetoną.
     Atlieka visą išankstinį apdorojimą: Sutrumpinkite, Pad, pridėkite specialius žetonus, kurių reikia jūsų modeliui.
오늘날 가장 많이 사용되는 토크나이저를 사용하여 새로운 어휘를 훈련하고 토큰화합니다.
     Rust 구현 덕분에 매우 빠릅니다(훈련 및 토큰화 모두). 서버 CPU에서 1GB의 텍스트를 토큰화하는 데 20초도 채 걸리지 않습니다.
     사용하기 쉽지만 매우 다양합니다.
     연구 및 생산용으로 설계되었습니다.
     정규화에는 정렬 추적이 포함됩니다. 주어진 토큰에 해당하는 원래 문장의 부분을 얻는 것은 항상 가능합니다.
     자르기, 채우기, 모델에 필요한 특수 토큰 추가 등 모든 사전 처리를 수행합니다.
     Азыркы эң көп колдонулган токенизаторлорду колдонуп, жаңы лексикаларды үйрөтүңүз жана токенизациялаңыз.
     Rust ишке ашыруунун аркасында абдан тез (окутуу жана токенизация). Сервердин процессорунда ГБ текстти белгилөө үчүн 20 секунддан аз убакыт кетет.
     Колдонуу оңой, бирок ошондой эле өтө ар тараптуу.
     Изилдөө жана өндүрүш үчүн иштелип чыккан.
     Нормалдаштыруу тегиздөөлөрдү көзөмөлдөө менен келет. Берилген белгиге туура келген баштапкы сүйлөмдүн бөлүгүн алуу ар дайым мүмкүн.
     Бардык алдын ала иштетүүнү аткарат: Кыскартыңыз, Pad, моделиңизге керектүү атайын белгилерди кошуңуз.
Träna nya ordfxrråd och tokenisera med dagens mest använda tokenizers.
     Extremt snabb (både träning och tokenisering), tack vare Rust-implementeringen. Det tar mindre än 20 sekunder att tokenisera en GB text på en servers CPU.
     Lätt att använda, men också extremt mångsidig.
     Designad fxr forskning och produktion.
     Normalisering kommer med anpassningsspårning. Det är alltid mxjligt att få den del av den ursprungliga meningen som motsvarar en given token.
     Gxr all fxrbearbetning: Truncate, Pad, lägg till de speciella tokens som din modell behxver.
                 out_q.device().is_meta() ? NULL : ((uint16_t*) out_q.data_ptr()) + c * columns,
            1,
            columns,
            qzero,
            maxq
        );

        adjust_error_row_cuda
        (
            (const float*) hessian_inv.data_ptr(),
            (float*) error.data_ptr(),
            (const float*) weights.data_ptr(),
            (const float*) quant.data_ptr(),
            c,
12345678912345678/92135678912315646845
364da
/................./////////
/* */"
" 2mkqwd "'2"'"'2'2'"@\\n\\n'#]

Let's write some cuneiform in a fixed-width font.
Normally, every character should line up with a character above.
Here: 𒈙. See how these characters don't align correctly?
Not only is it a very wide glyph, but its width is not even a multiple.
At least not in my font (Mac Safari 15.0).
But Ǆ is ok.


posted 14 years ago

    Mark post as helpful send pies Quote Report post to moderator 

Hi,
?
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
	
public static void main(String[] args) throws FileNotFoundException, IOException,
            UnsupportedRTFTemplate, Exception {
        System.setProperty("file.enconding", "ISO-8859-1");
        RTFTemplateBuilder builder = RTFTemplateBuilder.newRTFTemplateBuilder();
        RTFTemplate rtf = builder.newRTFTemplate();
        FileReader rtfSource = new FileReader("source.rtf");
        File rtfTarget = new File("target.rtf");
        copyFile(new File("source.rtf"), rtfTarget.getAbsolutePath());
        rtf.setTemplate(rtfSource);
        rtf.put("NAME", "Maiko Cezar");
        String x = new String("São Paulo");
        System.out.println(x);
        x = new String(x.getBytes(), "ISO-8859-1");
        System.out.println(x);
        rtf.put("CITY", x);
        rtf.merge(rtfTarget);
    }


my output
São Paulo
SÃ£o Paulo

And then I put this (SÃ£o Paulo) in the rtfTarget, but without successful. The "ISO-8859-1" encode it's the rtf native encode, isn't?

bb

𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂𒀂

😀 	😀 	😀 	— 	— 	— 	grinning face
2 	U+1F603 	😃 	😃 	😃 	😃 	😃 	😃 	grinning face with big eyes
3 	U+1F604 	😄 	😄 	😄 	😄 	— 	— 	grinning face with smiling eyes
4 	U+1F601 	😁 	😁 	😁 	😁 	😁 	😁 	beaming face with smiling eyes
5 	U+1F606 	😆 	😆 	😆 	— 	😆 	— 	grinning squinting face
6 	U+1F605 	😅 	😅 	😅 	— 	😅 	— 	grinning face with sweat
7 	U+1F923 	🤣 	🤣 	— 	— 	— 	— 	rolling on the floor laughing
8 	U+1F602 	😂 	😂 	😂 	😂 	— 	😂 	face with tears of joy
9 	U+1F642 	🙂 	🙂 	🙂 	— 	— 	— 	slightly smiling face
10 	U+1F643 	🙃 	🙃 	— 	— 	— 	— 	upside-down face
11 	U+1FAE0 	🫠 	🫠 	— 	— 	— 	— 	melting face
12 	U+1F609 	😉 	😉 	😉

😋 	😋 	😋 	— 	😋 	— 	face savoring food
25 	U+1F61B 	😛 	😛 	— 	— 	— 	— 	face with tongue
26 	U+1F61C 	😜 	😜 	😜 	😜 	😜 	😜 	winking face with tongue
27 	U+1F92A 	🤪 	🤪 	— 	— 	— 	— 	zany face
28 	U+1F61D 	😝 	😝 	😝 	😝 	— 	— 	squinting face with tongue
29 	U+1F911 	🤑 	🤑 	— 	— 	—

24 	U+1F60B 	😋 	😋 	😋 	— 	😋 	— 	face savoring food
25 	U+1F61B 	😛 	😛 	— 	— 	— 	— 	face with tongue
26 	U+1F61C 	😜 	😜 	😜 	😜 	😜 	😜 	winking face with tongue
27 	U+1F92A 	🤪 	🤪 	— 	— 	— 	— 	zany face

"""


for i in range(2000):
    p = random.randint(0, len(text) - 10)
    l = random.randint(0, len(text) // 2)

    print(".", end="")

    chunk = text[p:p+l]
    x = reference_tokenizer.encode(chunk, add_special_tokens = False)
    y = exl2_tokenizer.encode(chunk)[0]
    if x != y.tolist():
        print("dammit")

    y_ = exl2_tokenizer.decode(y)
    if y_ != chunk:
        print("curses")
