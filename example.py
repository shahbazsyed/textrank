from summa import fast_summarizer, summarizer
import glob

# documents = glob.glob('data/*.txt')
# print(documents)
# for doc in documents:
#     print(f'Processing {doc} ...')
#     out_path = 'summaries'+doc.replace('data','').replace('.txt','_ft_emb_summary.txt')
#     with open(doc, 'r') as inf, open(out_path, 'w') as out:
#         text = inf.read()
#         summary = summarizer.summarize(text,embedding_similarity=True, ratio=0.3)
#         if len(summary):
#             out.write(summary)
#         else:
#             print("Skipping empty summary")

text = """
The Charlie Hebdo cartoonists were smarter theologians than the jihadis.

Rather disturbingly, one word seems to connect the activity of the Paris terrorists and that of the Charlie Hebdo cartoonists: iconoclasm. I say disturbingly, because pointing out some common ground may be seen as blurring the crucial distinction between murderous bastards and innocent satirists. 

Nonetheless, it strikes me as fascinating that the cartoonists were profoundly iconoclastic in their constant ridiculing of religion (all religions, it must be noted) – yet it is precisely this same ancient tradition of iconoclasm that inspires Jews and Muslims to resist representational art and, in its most twisted pathological form, to attack the offices of a Paris magazine and slaughter those whose only weapon was the pen. So what’s the connection? 

In one sense an iconoclast is someone who refuses the established view of things, who kicks out against cherished beliefs and institutions. Which sounds pretty much like Charlie Hebdo. But the word iconoclast also describes those religious people who refuse and smash representational images, especially of the divine. The second of the Ten Commandments prohibits graven images – which is why there are no pictures of God in Judaism or Islam. And theologically speaking, the reason they are deeply suspicious of divine representation is because they fear that such representations of God might get confused for the real thing. The danger, they believe, is that we might end up overinvesting in a bad copy, something that looks a lot like what we might think of as god, but which, in reality, is just a human projection. So much better then to smash all representations of the divine. 

And yet this, of course, is exactly what Charlie Hebdo was doing. In the bluntest, rudest, most scatological and offensive of terms, Charlie Hebdo has been insisting that the images people worship are just human creations – bad and dangerous human creations. And in taking the piss out of such images, they actually exist in a tradition of religious iconoclasts going back as far as Abraham taking a hammer to his father’s statues. Both are attacks on representations of the divine. Which is why the terrorists, as well as being murderers, are theologically mistaken in thinking Charlie Hebdo is the enemy. For if God is fundamentally unrepresentable, then any representation of God is necessarily less than God and thus deserves to be fully and fearlessly attacked. And what better way of doing this than through satire, like scribbling a little moustache on a grand statue of God. 

It’s only some Christians that might demur from this logic. For the Christian story (ie Christmas) tells of God coming into the world as a human being. And in doing this the divine takes on a representational form. Jesus, says St Paul, is the image of the invisible God. Without this the whole western tradition of Catholic religious art and eastern tradition of icons is utterly inconceivable. 

So I can see how some Christians might find disrespectful images of God a la Charlie Hebdo insulting. After all, they believe it’s possible for images genuinely to capture something of God. But if images can’t capture God – and that’s the Jewish/Islamic position – then bring on the cartoonists. They should be encouraged to do their worst and disrupt our confidence that what we think of as God bears any resemblance to anything legitimately God-like. They are like Moses bringing down the golden calf. It is the ancient job of the iconoclast. 

But, of course, these terrorists weren’t really interested in theology. They thought that Charlie Hebdo’s cartoonists were insulting their human tribe, a tribe they called fellow Muslims. And maybe they were. But whatever else was happening, it was the atheist cartoonists who were performing the religious function and the apparently believing Muslims who had forgotten their deepest religious insights. For any representation of the divine that leads people to murder each other deserves the maximum possible disrespect. 
"""

ft_summary = fast_summarizer.summarize(text, ratio=0.4, weight_function="lexical_overlap")
summary = summarizer.summarize(text, ratio=0.4)
print("-"*50)
print(summary.replace('\n', ' '))
print('#' * 50)
print(ft_summary.replace('\n',' '))
print("-"*50)
        