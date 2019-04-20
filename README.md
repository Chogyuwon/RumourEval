# RumourEval
SemEval-2019 Task-7 : <https://competitions.codalab.org/competitions/19938>



### Welcome to RumourEval 2019!

The core mission is to automatically determine the veracity of rumours. The task falls into two parts; task A, in which responses to a rumourous post are classified according to stance, and task B, in which the statements themselves are classified for veracity. Each is described in more detail below.



#### Task A (SDQC)

Related to the objective of predicting a rumour's veracity, the first subtask will deal with the complementary objective of tracking how other sources orient to the accuracy of the rumourous story. A key step in the analysis of the surrounding discourse is to determine how other users in social media regard the rumour. 

We propose to tackle this analysis by looking at the replies to the post that presented the rumourous statement, i.e. the originating rumourous (source) post. 

We will provide participants with a tree-structured conversation formed of posts replying to the originating rumourous post, where each post presents its own type of support with regard to the rumour. 

We frame this in terms of supporting, denying, querying or commenting on (SDQC) the claim. 

Therefore, we introduce a subtask where the goal is to label the type of interaction between a given statement (rumourous post) and a reply post (the latter can be either direct or nested replies). 

Each tweet in the tree-structured thread will have to be categorized into one of the following <u>four categories</u>:



- Support (__S__): the author of the response supports the veracity of the rumour they are responding to.

- Deny (__D__): the author of the response denies the veracity of the rumour they are responding to.

- Query (__Q__): the author of the response asks for additional evidence in relation to the veracity of the rumour they are responding to.

- Comment (__C__): the author of the response makes their own comment without a clear contribution to assessing the veracity of the rumour they are responding to.



#### Task B (verification)

The goal of the second subtask is to predict the veracity of a given rumour. 

The rumour is presented as a post reporting or querying a claim but deemed unsubstantiated at the time of release. Given such a claim, and a set of other resources provided, systems should return a label describing the anticipated veracity of the rumour as true or false. 

The ground truth of this task is manually established by journalist and expert members of the team who identify official statements or other trustworthy sources of evidence that resolve the veracity of the given rumour. Additional context will be provided as input to veracity prediction systems; this context will consist of snapshots of relevant sources retrieved immediately before the rumour was reported, including a snapshot of an associated Wikipedia article, a Wikipedia dump, news articles from digital news outlets retrieved from NewsDiffs, as well as preceding tweets from the same event. 

Critically, no external resources may be used that contain information from after the rumour's resolution. 

To control this, we will specify precise versions of external information that participants may use. This is important to make sure we introduce time sensitivity into the task of veracity prediction. We take a simple approach to this task, using only true/false labels for rumours. 

In practice, however, many claims are hard to verify; for example, there were many rumours concerning Vladimir Putin's activities in early 2015, many wholly unsubstantiable. Therefore, we also expect systems to return a confidence value in the range of 0-1 for each rumour; if the rumour is unverifiable, a confidence of 0 should be returned.



#### Leaderboard

<strong>Latest news!</strong>&nbsp;The competition has ended. Thank you to all teams who showed interest and made submissions. The final leaderboard is below.



<center>
<table style="width: 80%;">
    <tbody>
        <tr style="font-weight: bold;">
            <td>User</td>
            <td>Verif</td>
            <td>RMSE</td>
            <td>SDQC</td>
        </tr>
        <tr>
            <td>quanzhi</td>
            <td>0.5765 (1)</td>
            <td>0.6078</td>
            <td>0.5776</td>
        </tr>
        <tr>
            <td>ukob-west</td>
            <td>0.2856 (2)</td>
            <td>0.7642</td>
            <td>0.3740</td>
        </tr>
        <tr>
            <td>sardar</td>
            <td>0.2620 (3)</td>
            <td>0.8012</td>
            <td>0.4352</td>
        </tr>
        <tr>
            <td>BLCU-nlp</td>
            <td>0.2525</td>
            <td>0.8179</td>
            <td>0.6187</td>
        </tr>
        <tr>
            <td>shaheyu</td>
            <td>0.2284</td>
            <td>0.8081</td>
            <td>0.3053</td>
        </tr>
        <tr>
            <td>ShivaliGoel</td>
            <td>0.2244</td>
            <td>0.8623</td>
            <td>0.3625</td>
        </tr>
        <tr>
            <td>mukundyr</td>
            <td>0.2244</td>
            <td>0.8623</td>
            <td>0.3404</td>
        </tr>
        <tr>
            <td>Xinthl</td>
            <td>0.2238</td>
            <td>0.8623</td>
            <td>0.2297</td>
        </tr>
        <tr>
            <td>lzr</td>
            <td>0.2238</td>
            <td>0.8678</td>
            <td>0.3404</td>
        </tr>
        <tr>
            <td>eebism</td>
            <td>0.1845</td>
            <td>0.7857</td>
            <td>0.2530</td>
        </tr>
        <tr>
            <td>Bilal.ghanem</td>
            <td>0.1996</td>
            <td>0.8264</td>
            <td>0.4895</td>
        </tr>
        <tr>
            <td>NimbusTwoThousand</td>
            <td>0.0950</td>
            <td>0.9148</td>
            <td>0.1272</td>
        </tr>
        <tr>
            <td>deanjjones</td>
            <td>0.0000</td>
            <td>0.0000</td>
            <td>0.3267</td>
        </tr>
        <tr>
            <td>jurebb</td>
            <td>0.0000</td>
            <td>0.0000</td>
            <td>0.3537</td>
        </tr>
        <tr>
            <td>z.zojaji</td>
            <td>0.0000</td>
            <td>0.0000</td>
            <td>0.3875</td>
        </tr>
        <tr>
            <td>lec-unifor</td>
            <td>0.0000</td>
            <td>0.0000</td>
            <td>0.4384</td>
        </tr>
        <tr>
            <td>magc</td>
            <td>0.0000</td>
            <td>0.0000</td>
            <td>0.3927</td>
        </tr>
        <tr>
            <td>Martin</td>
            <td>0.0000</td>
            <td>0.0000</td>
            <td>0.6067</td>
        </tr>
        <tr>
            <td>jacobvan</td>
            <td>0.0000</td>
            <td>0.0000</td>
            <td>0.4792</td>
        </tr>
        <tr>
            <td>wshuyi</td>
            <td>0.0000</td>
            <td>0.0000</td>
            <td>0.3699</td>
        </tr>
        <tr>
            <td>cjliux</td>
            <td>0.0000</td>
            <td>0.0000</td>
            <td>0.4298</td>
        </tr>
    </tbody>
</table>
</center>



You can still submit for your own experimentation purposes.

You can also join the <a href="https://groups.google.com/forum/#!forum/rumoureval">Google group</a> for the task, where you will find answers to your questions.



#### Organizers :

- Codalab lead and Reddit data: Genevieve Gorrell
- Twitter new (test) data: Ahmet Aker
- Danish and Russian data: Leon Derczynski
- Baseline: Elena Kochkina
- Advice and support from the rest of the team: Arkaitz Zubiaga, Maria Liakata, Kalina Bontcheva



# Dataset

You can download RumourEval Dataset from  [CodaLab](https://competitions.codalab.org/competitions/19938)









# RumourEval Papers




