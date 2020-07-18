import praw
import pandas as pd
import numpy as np
import sqlite3

class redditMethods:

    global comment2df
    def comment2df(comment):
        try:
            out = pd.DataFrame(data = {'author_fullname':comment.author_fullname,
                                               'comment':comment.body,
                                               'score':comment.score,
                                               'ups':comment.ups,
                                               'downs':comment.downs,
                                               'total_awards_received':comment.total_awards_received,
                                               'controversiality':comment.controversiality,
                                               'depth':comment.depth,
                                               'created':comment.created}, index = range(1))
        except:
            out = pd.DataFrame(data = {'author_fullname':None,'comment':None,'score':None,
                                       'ups':None,'downs':None,'total_awards_received':None,
                                       'controversiality':None,'depth':None,'created':None}, index = range(0))
        return out
    
    @staticmethod
    def getComments(x):
        x.comments.replace_more(limit=None)
        topComments = []
        for top_level_comment in x.comments:
            comment = comment2df(top_level_comment)
            topComments.append(comment)
        try:
            topComments = pd.concat(topComments)
            topComments.reset_index(drop = True, inplace = True)
        except:
            topComments = pd.DataFrame(data = {'author_fullname':None,'comment':None,'score':None,
                                               'ups':None,'downs':None,'total_awards_received':None,
                                               'controversiality':None,'depth':None,'created':None}, index = range(0))
        return topComments
            
    @staticmethod
    def getTopComments(x):
        x.comments.replace_more(limit=0)
        topComments = []
        for top_level_comment in x.comments:
            comment = comment2df(top_level_comment)
            topComments.append(comment)
        try:
            topComments = pd.concat(topComments)
            topComments.reset_index(drop = True, inplace = True)
        except:
            topComments = pd.DataFrame(data = {'author_fullname':None,'comment':None,'score':None,
                                               'ups':None,'downs':None,'total_awards_received':None,
                                               'controversiality':None,'depth':None,'created':None}, index = range(0))
        return topComments
        
    @staticmethod
    def getMedia(x):
        try:
            media = x.preview['images'][0]['source']['url']
        except:
            media = None
        return media
    
    @staticmethod
    def getSubmission(x):
        
        submission = [x.all_awardings,x.allow_live_comments,x.approved_by,x.archived,x.author,
                      x.author_flair_type,x.author_fullname,x.awarders,x.can_mod_post,x.category,
                      x.clicked,x.comment_sort, x.comments, x.content_categories,x.contest_mode,
                      x.created,x.created_utc,x.discussion_type,x.distinguished,x.domain,x.downs,
                      x.edited,x.fullname,x.gilded,x.hidden,x.hide_score,x.id,x.is_crosspostable,
                      x.is_meta,x.is_original_content,x.is_reddit_media_domain,x.is_robot_indexable,
                      x.is_self,x.is_video,x.likes,x.locked,x.media,x.media_only,x.name,x.num_comments,
                      x.num_crossposts,x.num_reports,x.over_18,x.parent_whitelist_status,x.permalink,
                      x.pinned,x.pwls,x.quarantine,x.report_reasons,x.score,x.secure_media,
                      x.selftext,x.send_replies,x.shortlink,x.spoiler,x.stickied,x.subreddit_id,
                      x.subreddit_name_prefixed,x.subreddit_subscribers,x.subreddit_type,
                      x.title,x.total_awards_received,x.ups,x.upvote,x.upvote_ratio,
                      x.url,x.view_count,x.visited,x.whitelist_status,x.wls]
        submission = [ y if y != [] else None for y in submission ]
        submissionCols = ['all_awardings','allow_live_comments','approved_by','archived','author',
                          'author_flair_type','author_fullname','awarders','can_mod_post','category',
                          'clicked','comment_sort', 'comments', 'content_categories','contest_mode',
                          'created','created_utc','discussion_type','distinguished','domain','downs',
                          'edited','fullname','gilded','hidden','hide_score','id','is_crosspostable',
                          'is_meta','is_original_content','is_reddit_media_domain','is_robot_indexable',
                          'is_self','is_video','likes','locked','media','media_only','name','num_comments',
                          'num_crossposts','num_reports','over_18','parent_whitelist_status','permalink',
                          'pinned','pwls','quarantine','report_reasons','score','secure_media',
                          'selftext','send_replies','shortlink','spoiler','stickied','subreddit_id',
                          'subreddit_name_prefixed','x.subreddit_subscribers','x.subreddit_type',
                          'title','total_awards_received','ups','upvote','upvote_ratio',
                          'url','view_count','visited','whitelist_status','wls']
        
        submission = pd.DataFrame(np.reshape(np.array(submission), (1,len(submission))), 
                                  columns = submissionCols, index = range(1))
        
        return submission

def obj2pd(obj, red):
    posts = []
    comments = []
    topComments = []
    
    for i in range(len(obj)):
        
        iPost = red.getSubmission(obj[i])
        iComments = red.getComments(obj[i])
        iTopComments = red.getTopComments(obj[i])
        iPicture = red.getMedia(obj[i])
        iPost['picture'] = iPicture
        iComments['id'] = iPost['id']
        iComments['subreddit_name_prefixed'] = iPost['subreddit_name_prefixed']
        iTopComments['id'] = iPost['id']
        iTopComments['subreddit_name_prefixed'] = iPost['subreddit_name_prefixed']
        
        posts.append(iPost)
        comments.append(iComments)
        topComments.append(iTopComments)
        
    posts = pd.concat(posts)
    comments = pd.concat(comments)
    topComments = pd.concat(topComments)
    
    return posts, comments, topComments

def posts2pd(obj, red):
    posts = []
    for i in range(len(obj)):
        try:
            iPost = red.getSubmission(obj[i])
            posts.append(iPost)
        except:
            pass
    posts = pd.concat(posts)
    return posts

if __name__ == '__main__':

    reddit = praw.Reddit(client_id='<YOUR_CLIENT_ID>',
                         client_secret='<YOUR_CLIENT_SECRET>',
                         password='<YOUR_PASSWORD>',
                         user_agent='<YOUR_UA>',
                         username='<YOUR_USERNAME>')
    
    try: 
        print(reddit.user.me())
    except:
        print('ruh roh')
        
    red = redditMethods()
    
    subreddits = ['introvert','extroverts','istj','istp','isfj','isfp',
                  'infj','infp','intj','intp','estp','estj2','esfp',
                  'esfj','enfp','enfj','entp','entj']
    
    allPosts = []
    #sort types: controversial, gilded, hot, new, rising, top
    for subreddit in subreddits:
        try:
            obj = []
            hot = list(reddit.subreddit(subreddit).hot(limit=1000))
            new = list(reddit.subreddit(subreddit).new(limit=1000))
            rising = list(reddit.subreddit(subreddit).rising(limit=1000))
            top = list(reddit.subreddit(subreddit).top(limit=1000))
            controversial = list(reddit.subreddit(subreddit).controversial(limit=1000))
            gilded = list(reddit.subreddit(subreddit).gilded(limit=1000))
            obj.extend(hot)
            obj.extend(new)
            obj.extend(rising)
            obj.extend(top)
            obj.extend(controversial)
            obj.extend(gilded)
            obj = list(obj) #keep in memory
            res = posts2pd(obj, red)
            allPosts.append(res)
        except:
            pass
            
    allPosts = pd.concat(allPosts)
    allPosts.drop(['all_awardings','upvote','media','secure_media'], axis = 1, inplace = True)
    allPosts[['author','comments']] = allPosts[['author','comments']].astype(str)
    
    conn = sqlite3.connect('reddit.sqlite')
    allPosts.to_sql('data', conn, if_exists = 'append', index = False)