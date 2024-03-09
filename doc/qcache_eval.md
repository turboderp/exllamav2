
## Q4 and FP8 cache evaluation

(More text coming soon.)

The tl;dr:

- Both Q4 and FP8 cache modes add very little loss compared to FP16
- Q4 is, perhaps suprisingly, more accurate than FP8
- Any loss of performance from Q4 on HumanEval is within the margin of error
- For summarizing long contexts there is no clearly discernible effect on the output 

### Perplexity

Token-level perplexity tests for various full-precision and quantized models using FP16, FP8 and Q4 cache
modes. Dataset is The Pile, 10 rows of 512 tokens per test. 

Model	| Precision	| FP16 cache	| FP8 cache	| Q4 cache
--------|-----------|---------------|-----------|---------
Mistral 7B Instruct	| 3.0 bpw	| 13.33	| 13.43	| 13.41
--	| 3.5 bpw	| 13.07	| 13.14	| 13.12
--	| 4.0 bpw	| 12.90	| 12.90	| 12.90
--	| 5.0 bpw	| 12.73	| 12.73	| 12.75
--	| 6.0 bpw	| 12.73	| 12.75	| 12.74
--	| FP16	| 12.69	| 12.71	| 12.72
Mixtral 8x7B	| 3.5 bpw	| 10.27	| 10.41	| 10.39
--	| 4.0 bpw	| 10.09	| 10.26	| 10.23
--	| 5.0 bpw	| 10.02	| 10.16	| 10.15
Llama2 7B	| 4.0 bpw	| 11.43	| 11.92	| 11.74
--	| 5.0 bpw	| 11.13	| 11.40	| 11.31
--	| FP16	| 10.91	| 11.24	| 11.16


### HumanEval

The following are HumanEval tests on various full-precision and quantized models using FP16 and Q4 cache,
respectively. Number of samples per task is limited to 10 (still giving 39360 completions in total produced
over about 24 hours.)

#### pass@1 

Model |	Precision	| FP16 cache  |	Q4 cache	| diff
------|-------------|-------------|-------------|-------
Mistral 7B Instruct	| 3.0 bpw	| 21.8%	| 20.9%	| -0.9%
--	| 3.5 bpw	| 23.1%	| 23.3%	| +0.2%
--	| 4.0 bpw	| 23.3%	| 24.0%	| +0.7%
--	| 5.0 bpw	| 25.1%	| 27.3%	| +2.2%
--	| 6.0 bpw	| 25.3%	| 26.0%	| +0.7%
--	| FP16	| 25.1%	| 26.0%	| +0.9%
Gemma 7B 	| 3.0 bpw	| 11.9%	| 11.9%	|
-- 	| 4.0 bpw	| 29.7%	| 29.8%	| +0.1%
-- 	| 5.0 bpw	| 31.1%	| 31.2%	| +0.1%
-- 	| 6.0 bpw	| 31.7%	| 32.0%	| +0.3%
-- 	| FP16	| 30.4%	| 32.0%	| +1.6%
Mixtral 8x7B	| 4.0 bpw	| 33.5%	| 31.3%	| -2.2%

#### pass@10

Model |	Precision	| FP16 cache  |	Q4 cache	| diff
------|-------------|-------------|-------------|-------
Mistral 7B Instruct	| 3.0 bpw	| 43.3%	| 47.0%	| +3.7%
--	| 3.5 bpw	| 48.2%	| 53.0%	| +4.8%
--	| 4.0 bpw	| 45.7%	| 48.2%	| +2.5%
--	| 5.0 bpw	| 51.2%	| 51.2%	| 
--	| 6.0 bpw	| 50.6%	| 51.2%	| +0.6%
--	| FP16	| 49.4%	| 53.0%	| +3.6%
Gemma 7B 	|	3.0 bpw	| 28.7%	| 29.9%	| +1.2%
-- 	|	4.0 bpw	| 59.8%	| 57.9%	| -1.9%
-- 	|	5.0 bpw	| 64.0%	| 62.8%	| -1.2%
-- 	| 	6.0 bpw	| 61.5%	| 65.2%	| +3.7%
-- 	|	FP16	| 64.0%	| 60.9%	| -3.1%
Mixtral 8x7B	|	4.0 bpw	| 59.8%	| 62.8%	| +3.0%

### Summarization

Here are summaries of [a Guardian article](https://www.theguardian.com/world/2024/mar/05/it-was-so-wrong-why-were-so-many-people-imprisoned-over-one-protest-in-bristol)
published in March 2024, totaling 8879 tokens. Summaries are produced by Miqu-70B at 4.25 bpw using default
settings in ExUI and the prompt `### Story summary:`

<details>
  <summary><b>FP16 cache, attempt 1</b></summary>
On 21 March 2021, Fleur Moody was knocked unconscious during a protest in Bristol against the police, crime, sentencing and courts bill. She was later charged with riot - a charge that required senior approval within the Crown Prosecution Service (CPS) and punishable with up to 10 years in prison. To date, 38 people have been charged with riot for their involvement in the Bristol protests, including Moody. Nineteen have been convicted, with sentences ranging from three to six years. Nine protesters are now in prison, and arrests are still being made. More people, and more women, have been imprisoned for rioting during one day in Bristol than in any other protest-related disorder since at least the 1980s. Evidence that has emerged during the protesters' trials suggests a different story: first, that many of the accused were, in fact, acting in self-defence against police aggression; and second, that the unprecedented deployment of riot charges may have had less to do with the severity of the violence during the protest than a politically driven crackdown on the right to protest.<br><br>
</details>
<details>
  <summary><b>FP16 cache, attempt 2</b></summary>
In March 2021, a protest against the police, crime, sentencing and courts bill in Bristol turned violent, resulting in the arrest of 38 people who were charged with riot. Of those, 19 have been convicted with sentences ranging from three to six years, and nine are currently in prison. This is more people and more women who have been imprisoned for rioting during one day in Bristol than in any other protest-related disorder since at least the 1980s. However, evidence that has emerged during the protesters' trials suggests that many of them were acting in self-defense against police aggression, and that the unprecedented deployment of riot charges may have had less to do with the severity of the violence during the protest than a politically driven crackdown on the right to protest.<br><br>
</details>
<details>
  <summary><b>FP16 cache, attempt 3</b></summary>
In March 2021, thousands of people gathered in Bristol to protest against the police, crime, sentencing and courts bill. The protest turned violent, with police and protesters clashing in the streets. In the aftermath, 38 people were charged with riot, a crime that carries a maximum sentence of 10 years in prison. To date, 19 people have been convicted, with sentences ranging from three to six years. The severity of the charges and sentences has raised questions about the right to protest and the role of the police in maintaining order.<br><br>
</details>

<details>
  <summary><b>FP8 cache, attempt 1</b></summary>
In March 2021, 38 people were charged with riot after a protest in Bristol against the police, crime, sentencing and courts bill. The convictions have raised concerns about the right to protest in the UK. The defendants, including Fleur Moody, were charged with riot after a protest in Bristol against the police, crime, sentencing and courts bill. Moody was knocked unconscious by police during the protest. The convictions have raised concerns about the right to protest in the UK.<br><br>
</details>
<details>
  <summary><b>FP8 cache, attempt 2</b></summary>
In March 2021, a protest against the police, crime, sentencing and courts bill in Bristol turned violent, resulting in 38 people being charged with riot. Since then, 19 have been convicted, with sentences ranging from three to six years. This is an unprecedented number of riot convictions for a single day of protest in the UK. The severity of the charges and sentences has raised questions about the police response to the protest and the role of political pressure in the prosecutions. Some defendants have claimed they were acting in self-defense against police aggression, and evidence has emerged that supports their claims. The case raises concerns about the right to protest and the balance between public order and civil liberties.<br><br>
</details>
<details>
  <summary><b>FP8 cache, attempt 3</b></summary>
The story begins with Fleur Moody, a 26-year-old woman who joined a protest against the Police, Crime, Sentencing and Courts Bill in Bristol on 21 March 2021. During the protest, Moody was knocked unconscious by a police officer. She was later charged with riot and pleaded guilty to the lesser charge of affray, receiving a suspended sentence.

The article then explores the context of the protest, which was in response to proposed laws that would give the police extensive new powers to clamp down on protests. The protest turned violent, with clashes between protesters and police resulting in injuries on both sides.

The article also highlights the unusually severe crackdown that followed the protest, with 38 people being charged with riot, a charge that carries a maximum sentence of 10 years in prison. Of those charged, 19 have been convicted, with sentences ranging from three to six years. Nine protesters are currently in prison.

The article raises questions about the excessive use of force by the police during the protest and the lack of accountability for police actions.<br><br>
</details>
<details>
  <summary><b>Q4 cache, attempt 1</b></summary>
On 21 March 2021, Fleur Moody, 26, joined a march protesting against proposed laws that would give the police extensive new powers to clamp down on protests. She was knocked unconscious by police officers during the protest and later charged with riot. To date, 38 people have been charged with riot for their involvement in the Bristol protests, including Moody. Nineteen have been convicted, with sentences ranging from three to six years. Nine protesters are now in prison, and arrests are still being made. The unprecedented deployment of riot charges may have had less to do with the severity of the violence during the protest than a politically driven crackdown on the right to protest.<br><br>
</details>
<details>
  <summary><b>Q4 cache, attempt 2</b></summary>
Fleur Moody was one of 38 people charged with riot for their involvement in the Bristol protests against the police, crime, sentencing and courts bill in March 2021. Of the 19 who have been convicted, nine are in prison, and arrests are still being made. The severity of the state's response has been attributed to the violent extremists who allegedly hijacked the protest. However, evidence from the trials suggests that many of the accused were acting in self-defence against police aggression. The unprecedented deployment of riot charges may have had more to do with a politically driven crackdown on the right to protest than the severity of the violence during the protest.<br><br>
</details>
<details>
  <summary><b>Q4 cache, attempt 3</b></summary>
In March 2021, Fleur Moody joined a protest in Bristol against proposed laws that would give the police extensive new powers to clamp down on protests. During the protest, Moody was struck on the head by a police officer and knocked unconscious. She was later charged with riot, despite having no memory of the incident. To date, 38 people have been charged with riot for their involvement in the Bristol protests, and 19 have been convicted, with sentences ranging from three to six years. This is the largest number of people to be imprisoned for rioting during one day in Bristol since at least the 1980s.<br><br>
</details>

Here are summaries of [another article](https://www.theguardian.com/books/2023/nov/09/andrew-wylie-agency-days-of-the-jackal-serious-literature-big-business),
published November 2023, totaling 11975 tokens:

<details>
  <summary><b>FP16 cache, attempt 1</b></summary>
In 2008, Andrew Wylie, the world’s most renowned literary agent, began plotting to take over the Chinese market after a bidding war broke out among Chinese publishers for the collected works of Jorge Luis Borges. His plan was to sell Henry Kissinger’s book On China to the Chinese market first, an unprecedented tactic for a book by a famous American author. Kissinger quickly became a client of The Wylie Agency, and the new book was sold to a Chinese publisher for more than $1m. This established Wylie’s authority in the Chinese market, and he has since achieved seven-figure deals in China for the works of authors such as Milan Kundera and Philip K Dick.<br><br>
</details>
<details>
  <summary><b>FP16 cache, attempt 2</b></summary>
Andrew Wylie, the world’s most renowned literary agent, has reshaped the business of publishing in profound and controversial ways over the past four decades. He has helped to define the global literary canon while making many great writers famous and rich. But his critics argue that he has hastened the demise of the literary culture he claims to defend. Wylie is largely untroubled by such criticisms and is currently focused on making deals in China. He has been known as The Jackal since 1995 when he lured Martin Amis away from another agent with a pledge to sell his novel for £500,000. Wylie’s ferocity for business is matched by his passion for literature, and he has been a champion of highbrow books and unabashed commerce. His clients include Nobel laureates and some of the most revered writers in the world.<br><br>
</details>
<details>
  <summary><b>FP16 cache, attempt 3</b></summary>
Andrew Wylie, the world’s most renowned literary agent, has reshaped the business of publishing in profound and controversial ways over the past four decades. He has represented many great writers and helped to define the global literary canon, but some critics argue that he has hastened the demise of literary culture. Wylie is currently focused on expanding his business in China, where he aims to establish his authority by representing high-profile authors like Henry Kissinger. He has been successful in achieving seven-figure deals for his clients in the Chinese market.<br><br>
</details>
<details>
  <summary><b>FP8 cache, attempt 1</b></summary>
Andrew Wylie is a literary agent who has represented some of the world's most renowned authors, including Philip Roth, Salman Rushdie, and Kenzaburō Ōe. He is known for his aggressive negotiating style and his ability to secure high advances for his clients. Wylie has been credited with reshaping the business of publishing and has been a champion of highbrow books and unabashed commerce. However, he has also been criticized for hastening the demise of literary culture. Wylie is now turning his attention to the Chinese market, where he has achieved seven-figure deals for the works of authors such as Milan Kundera and Philip K Dick.<br><br>
</details>
<details>
  <summary><b>FP8 cache, attempt 2</b></summary>
Andrew Wylie is a literary agent who has represented a wide range of prominent authors, including Salman Rushdie, Philip Roth, and Martin Amis. He is known for his aggressive negotiation tactics and his ability to secure high advances for his clients. Wylie has been credited with raising the value of literary properties and helping to establish the careers of many successful authors. However, he has also been criticized for his aggressive approach and for contributing to the commercialization of literature. Despite these criticisms, Wylie remains one of the most influential figures in the publishing industry.<br><br>
</details>
<details>
  <summary><b>FP8 cache, attempt 3</b></summary>
Andrew Wylie is 76 years old and is one of the most influential literary agents in the world. He has represented some of the most renowned authors, including Jorge Luis Borges, Philip Roth, and Salman Rushdie. Wylie is known for his aggressive approach to negotiating deals and has been credited with transforming the business of publishing. He has been criticized for his tactics, but he has also been praised for his commitment to literary quality and his ability to secure lucrative deals for his clients. Wylie's agency, The Wylie Agency, represents over 1,300 authors from around the world and has offices in New York, London, and Madrid.<br><br>
</details>
<details>
  <summary><b>Q4 cache, attempt 1</b></summary>
Andrew Wylie, the world’s most renowned literary agent, has reshaped the business of publishing in profound and controversial ways over the past four decades. He has represented many great writers and helped to define the global literary canon, but some critics argue that he has hastened the demise of literary culture. Wylie is currently focused on expanding his business in China, where he has achieved seven-figure deals for the works of authors such as Milan Kundera and Philip K Dick. He is known for his aggressive business tactics and his ability to command high prices for his clients’ works. However, some in the publishing industry argue that Wylie has coarsened the industry and eroded the broader ecosystem of literary publishing.<br><br>
</details>
<details>
  <summary><b>Q4 cache, attempt 2</b></summary>
Andrew Wylie, 76, is the world’s most renowned literary agent, representing an astonishing number of the world’s most revered writers, as well as the estates of many late authors who have become required reading almost everywhere. The agency’s list of more than 1,300 clients includes Saul Bellow, Joseph Brodsky, Albert Camus, Bob Dylan, Louise Glück, Yasunari Kawabata, Czesław Miłosz, VS Naipaul, Kenzaburō Ōe, Orhan Pamuk, José Saramago and Mo Yan – and those are just the ones who have won the Nobel prize. It also includes the Royal Shakespeare Company and contemporary luminaries such as Chimamanda Ngozi Adichie, Karl Ove Knausgård, Rachel Cusk, Deborah Levy and Sally Rooney.<br><br>
</details>
<details>
  <summary><b>Q4 cache, attempt 3</b></summary>
Andrew Wylie, the most famous literary agent in the world, has spent decades reshaping the business of publishing. He has represented the works of Borges, Calvino, Dylan, Knausgård, Levy, Naipaul, Pamuk, Rooney and many others. He has also helped to define the global literary canon. Critics argue that he has hastened the demise of literary culture, but Wylie is untroubled by such criticisms. He is currently focused on making deals in China, where he sees a huge potential market for foreign literary works. In 2008, a bidding war broke out among Chinese publishers for the collected works of Borges, and Wylie decided to try to dictate the value of other foreign works in the Chinese market. He has since achieved seven-figure deals in China for the works of authors such as Kundera and Philip K Dick.<br><br>
</details>




