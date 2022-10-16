import numpy as np
from scipy.stats import sem

########################################################################################################################

# Juror 1

j1 = np.array([[[3, 5, 5], # sample 1: cit: c1,c2,c3
                [5, 4, 5], # sample 1: c-s: c1,c2,c3
                [4, 4, 2], # sample 1: t-a: c1,c2,c3
                [4, 3, 2]], # sample 1: i-e: c1,c2,c3

               [[5, 5, 5], # sample 2: cit: c1,c2,c3
                [5, 5, 3], # sample 2: c-s: c1,c2,c3
                [5, 5, 2], # sample 2: t-a: c1,c2,c3
                [2, 4, 1]], # sample 2: i-e: c1,c2,c3
                      
               [[4, 3, 5], # sample 3: cit: c1,c2,c3
                [5, 5, 4], # sample 3: c-s: c1,c2,c3
                [5, 5, 3], # sample 3: t-a: c1,c2,c3
                [4, 4, 1]], # sample 3: i-e: c1,c2,c3

               [[2, 2, 5], # sample 4: cit: c1,c2,c3
                [5, 5, 5], # sample 4: c-s: c1,c2,c3
                [3, 4, 3], # sample 4: t-a: c1,c2,c3
                [5, 3, 2]], # sample 4: i-e: c1,c2,c3
                      
               [[4, 3, 5], # sample 5: cit: c1,c2,c3
                [5, 2, 5], # sample 5: c-s: c1,c2,c3
                [4, 4, 2], # sample 5: t-a: c1,c2,c3
                [2, 5, 2]], # sample 5: i-e: c1,c2,c3

               [[4, 4, 5], # sample 6: cit: c1,c2,c3
                [5, 4, 5], # sample 6: c-s: c1,c2,c3
                [4, 2, 3], # sample 6: t-a: c1,c2,c3
                [5, 4, 3]], # sample 6: i-e: c1,c2,c3
                      
               [[5, 5, 5], # sample 7: cit: c1,c2,c3
                [4, 4, 5], # sample 7: c-s: c1,c2,c3
                [4, 3, 2], # sample 7: t-a: c1,c2,c3
                [2, 4, 1]], # sample 7: i-e: c1,c2,c3

               [[4, 5, 5], # sample 8: cit: c1,c2,c3
                [4, 4, 4], # sample 8: c-s: c1,c2,c3
                [5, 3, 2], # sample 8: t-a: c1,c2,c3
                [5, 4, 1]], # sample 8: i-e: c1,c2,c3
                      
               [[4, 5, 5], # sample 9: cit: c1,c2,c3
                [4, 5, 4], # sample 9: c-s: c1,c2,c3
                [3, 4, 2], # sample 9: t-a: c1,c2,c3
                [5, 3, 1]], # sample 9: i-e: c1,c2,c3

               [[2, 4, 5], # sample 10: cit: c1,c2,c3
                [5, 5, 4], # sample 10: c-s: c1,c2,c3
                [4, 3, 2], # sample 10: t-a: c1,c2,c3
                [2, 4, 1]], # sample 10: i-e: c1,c2,c3
                      
               [[2, 4, 5], # sample 11: cit: c1,c2,c3
                [4, 4, 5], # sample 11: c-s: c1,c2,c3
                [3, 2, 1], # sample 11: t-a: c1,c2,c3
                [2, 4, 1]], # sample 11: i-e: c1,c2,c3

               [[3, 2, 5], # sample 12: cit: c1,c2,c3
                [5, 4, 4], # sample 12: c-s: c1,c2,c3
                [5, 3, 2], # sample 12: t-a: c1,c2,c3
                [4, 4, 1]], # sample 12: i-e: c1,c2,c3
                      
               [[2, 4, 5], # sample 13: cit: c1,c2,c3
                [4, 4, 4], # sample 13: c-s: c1,c2,c3
                [2, 5, 3], # sample 13: t-a: c1,c2,c3
                [2, 3, 3]], # sample 13: i-e: c1,c2,c3

               [[5, 4, 5], # sample 14: cit: c1,c2,c3
                [5, 4, 5], # sample 14: c-s: c1,c2,c3
                [4, 4, 3], # sample 14: t-a: c1,c2,c3
                [2, 4, 1]], # sample 14: i-e: c1,c2,c3
                      
               [[5, 4, 5], # sample 15: cit: c1,c2,c3
                [5, 3, 4], # sample 15: c-s: c1,c2,c3
                [2, 4, 1], # sample 15: t-a: c1,c2,c3
                [3, 4, 1]], # sample 15: i-e: c1,c2,c3

               [[2, 4, 5], # sample 16: cit: c1,c2,c3
                [5, 3, 4], # sample 16: c-s: c1,c2,c3
                [5, 2, 4], # sample 16: t-a: c1,c2,c3
                [4, 4, 1]], # sample 16: i-e: c1,c2,c3
                      
               [[3, 2, 5], # sample 17: cit: c1,c2,c3
                [4, 5, 4], # sample 17: c-s: c1,c2,c3
                [3, 4, 2], # sample 17: t-a: c1,c2,c3
                [4, 4, 5]], # sample 17: i-e: c1,c2,c3

               [[5, 3, 5], # sample 18: cit: c1,c2,c3
                [5, 4, 5], # sample 18: c-s: c1,c2,c3
                [4, 4, 3], # sample 18: t-a: c1,c2,c3
                [4, 5, 3]], # sample 18: i-e: c1,c2,c3
                      
               [[2, 4, 5], # sample 19: cit: c1,c2,c3
                [5, 3, 5], # sample 19: c-s: c1,c2,c3
                [2, 3, 2], # sample 19: t-a: c1,c2,c3
                [3, 5, 2]], # sample 19: i-e: c1,c2,c3

               [[4, 4, 5], # sample 20: cit: c1,c2,c3
                [4, 5, 5], # sample 20: c-s: c1,c2,c3
                [2, 4, 2], # sample 20: t-a: c1,c2,c3
                [2, 5, 2]], # sample 20: i-e: c1,c2,c3
                      
               [[2, 4, 5], # sample 21: cit: c1,c2,c3
                [4, 4, 4], # sample 21: c-s: c1,c2,c3
                [2, 3, 4], # sample 21: t-a: c1,c2,c3
                [5, 4, 1]], # sample 21: i-e: c1,c2,c3

               [[4, 5, 5], # sample 22: cit: c1,c2,c3
                [5, 4, 5], # sample 22: c-s: c1,c2,c3
                [5, 4, 4], # sample 22: t-a: c1,c2,c3
                [2, 4, 3]], # sample 22: i-e: c1,c2,c3
                      
               [[1, 4, 5], # sample 23: cit: c1,c2,c3
                [5, 3, 5], # sample 23: c-s: c1,c2,c3
                [2, 4, 2], # sample 23: t-a: c1,c2,c3
                [1, 4, 2]], # sample 23: i-e: c1,c2,c3

               [[4, 4, 5], # sample 24: cit: c1,c2,c3
                [4, 4, 5], # sample 24: c-s: c1,c2,c3
                [4, 4, 3], # sample 24: t-a: c1,c2,c3
                [4, 3, 2]], # sample 24: i-e: c1,c2,c3
                      
               [[4, 3, 5], # sample 25: cit: c1,c2,c3
                [5, 3, 5], # sample 25: c-s: c1,c2,c3
                [4, 4, 3], # sample 25: t-a: c1,c2,c3
                [3, 4, 2]], # sample 25: i-e: c1,c2,c3

               [[3, 4, 5], # sample 26: cit: c1,c2,c3
                [3, 4, 5], # sample 26: c-s: c1,c2,c3
                [3, 1, 5], # sample 26: t-a: c1,c2,c3
                [2, 2, 4]], # sample 26: i-e: c1,c2,c3                

               [[4, 3, 5], # sample 27: cit: c1,c2,c3
                [4, 3, 4], # sample 27: c-s: c1,c2,c3
                [4, 4, 3], # sample 27: t-a: c1,c2,c3
                [4, 3, 4]], # sample 27: i-e: c1,c2,c3
                      
               [[3, 2, 5], # sample 28: cit: c1,c2,c3,
                [4, 4, 5], # sample 28: c-s: c1,c2,c3
                [3, 3, 4], # sample 28: t-a: c1,c2,c3
                [4, 4, 3]], # sample 28: i-e: c1,c2,c3

               [[3, 4, 5], # sample 29: cit: c1,c2,c3
                [4, 3, 4], # sample 29: c-s: c1,c2,c3
                [2, 2, 1], # sample 29: t-a: c1,c2,c3
                [4, 3, 1]], # sample 29: i-e: c1,c2,c3
                      
               [[3, 3, 5], # sample 30: cit: c1,c2,c3
                [4, 3, 4], # sample 30: c-s: c1,c2,c3
                [4, 3, 4], # sample 30: t-a: c1,c2,c3
                [3, 4, 3]] # sample 30: i-e: c1,c2,c3
            ])

j1_confidence = np.array([[[False, False, False], # sample 1: cit: c1,c2,c3
                           [False, False, False], # sample 1: c-s: c1,c2,c3
                           [False, False, False], # sample 1: t-a: c1,c2,c3
                           [False, False, False]], # sample 1: i-e: c1,c2,c3

                          [[False, False, False], # sample 2: cit: c1,c2,c3
                           [False, False, False], # sample 2: c-s: c1,c2,c3
                           [False, False, True], # sample 2: t-a: c1,c2,c3
                           [False, False, False]], # sample 2: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 3: cit: c1,c2,c3
                           [False, False, True], # sample 3: c-s: c1,c2,c3
                           [False, False, True], # sample 3: t-a: c1,c2,c3
                           [False, False, False]], # sample 3: i-e: c1,c2,c3

                          [[False, False, False], # sample 4: cit: c1,c2,c3
                           [False, False, False], # sample 4: c-s: c1,c2,c3
                           [False, False, False], # sample 4: t-a: c1,c2,c3
                           [False, False, False]], # sample 4: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 5: cit: c1,c2,c3
                           [False, False, False], # sample 5: c-s: c1,c2,c3
                           [False, False, False], # sample 5: t-a: c1,c2,c3
                           [False, False, False]], # sample 5: i-e: c1,c2,c3

                          [[False, False, False], # sample 6: cit: c1,c2,c3
                           [False, False, False], # sample 6: c-s: c1,c2,c3
                           [False, False, True], # sample 6: t-a: c1,c2,c3
                           [False, False, False]], # sample 6: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 7: cit: c1,c2,c3
                           [False, False, False], # sample 7: c-s: c1,c2,c3
                           [False, False, False], # sample 7: t-a: c1,c2,c3
                           [False, False, True]], # sample 7: i-e: c1,c2,c3

                          [[False, False, False], # sample 8: cit: c1,c2,c3
                           [False, False, False], # sample 8: c-s: c1,c2,c3
                           [False, False, True], # sample 8: t-a: c1,c2,c3
                           [False, False, False]], # sample 8: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 9: cit: c1,c2,c3
                           [False, False, False], # sample 9: c-s: c1,c2,c3
                           [False, False, False], # sample 9: t-a: c1,c2,c3
                           [False, False, True]], # sample 9: i-e: c1,c2,c3

                          [[False, False, False], # sample 10: cit: c1,c2,c3
                           [False, False, True], # sample 10: c-s: c1,c2,c3
                           [False, False, False], # sample 10: t-a: c1,c2,c3
                           [False, False, False]], # sample 10: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 11: cit: c1,c2,c3
                           [False, False, False], # sample 11: c-s: c1,c2,c3
                           [False, False, False], # sample 11: t-a: c1,c2,c3
                           [False, False, False]], # sample 11: i-e: c1,c2,c3

                          [[False, False, False], # sample 12: cit: c1,c2,c3
                           [False, False, False], # sample 12: c-s: c1,c2,c3
                           [False, False, False], # sample 12: t-a: c1,c2,c3
                           [False, False, False]], # sample 12: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 13: cit: c1,c2,c3
                           [False, False, False], # sample 13: c-s: c1,c2,c3
                           [False, False, True], # sample 13: t-a: c1,c2,c3
                           [False, False, True]], # sample 13: i-e: c1,c2,c3

                          [[False, False, False], # sample 14: cit: c1,c2,c3
                           [False, False, False], # sample 14: c-s: c1,c2,c3
                           [False, False, False], # sample 14: t-a: c1,c2,c3
                           [False, False, False]], # sample 14: i-e: c1,c2,c3
                               
                          [[False, False, False], # sample 15: cit: c1,c2,c3
                           [False, False, False], # sample 15: c-s: c1,c2,c3
                           [False, False, False], # sample 15: t-a: c1,c2,c3
                           [False, False, False]], # sample 15: i-e: c1,c2,c3

                          [[False, False, False], # sample 16: cit: c1,c2,c3
                           [False, False, False], # sample 16: c-s: c1,c2,c3
                           [False, False, False], # sample 16: t-a: c1,c2,c3
                           [False, False, False]], # sample 16: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 17: cit: c1,c2,c3
                           [False, False, False], # sample 17: c-s: c1,c2,c3
                           [False, False, False], # sample 17: t-a: c1,c2,c3
                           [False, False, False]], # sample 17: i-e: c1,c2,c3

                          [[False, False, False], # sample 18: cit: c1,c2,c3
                           [False, False, False], # sample 18: c-s: c1,c2,c3
                           [False, False, False], # sample 18: t-a: c1,c2,c3
                           [False, False, False]], # sample 18: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 19: cit: c1,c2,c3
                           [False, False, False], # sample 19: c-s: c1,c2,c3
                           [False, False, False], # sample 19: t-a: c1,c2,c3
                           [False, False, False]], # sample 19: i-e: c1,c2,c3

                          [[False, False, False], # sample 20: cit: c1,c2,c3
                           [False, False, False], # sample 20: c-s: c1,c2,c3
                           [False, False, False], # sample 20: t-a: c1,c2,c3
                           [False, False, False]], # sample 20: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 21: cit: c1,c2,c3
                           [False, False, False], # sample 21: c-s: c1,c2,c3
                           [False, False, False], # sample 21: t-a: c1,c2,c3
                           [False, False, False]], # sample 21: i-e: c1,c2,c3

                          [[False, False, False], # sample 22: cit: c1,c2,c3
                           [False, False, False], # sample 22: c-s: c1,c2,c3
                           [False, False, False], # sample 22: t-a: c1,c2,c3
                           [False, False, True]], # sample 22: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 23: cit: c1,c2,c3
                           [False, False, False], # sample 23: c-s: c1,c2,c3
                           [False, False, True], # sample 23: t-a: c1,c2,c3
                           [False, False, True]], # sample 23: i-e: c1,c2,c3

                          [[False, False, False], # sample 23: cit: c1,c2,c3
                           [False, False, False], # sample 24: c-s: c1,c2,c3
                           [False, False, False], # sample 24: t-a: c1,c2,c3
                           [False, False, True]], # sample 24: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 25: cit: c1,c2,c3
                           [False, False, False], # sample 25: c-s: c1,c2,c3
                           [False, False, False], # sample 25: t-a: c1,c2,c3
                           [False, False, False]], # sample 25: i-e: c1,c2,c3

                          [[False, False, False], # sample 26: cit: c1,c2,c3
                           [False, False, False], # sample 26: c-s: c1,c2,c3
                           [False, False, False], # sample 26: t-a: c1,c2,c3
                           [False, False, False]], # sample 26: i-e: c1,c2,c3                

                          [[False, False, False], # sample 27: cit: c1,c2,c3
                           [False, False, False], # sample 27: c-s: c1,c2,c3
                           [False, False, True], # sample 27: t-a: c1,c2,c3
                           [False, False, False]], # sample 27: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 28: cit: c1,c2,c3,
                           [False, False, False], # sample 28: c-s: c1,c2,c3
                           [False, False, False], # sample 28: t-a: c1,c2,c3
                           [False, False, True]], # sample 28: i-e: c1,c2,c3

                          [[False, False, False], # sample 29: cit: c1,c2,c3
                           [False, False, False], # sample 29: c-s: c1,c2,c3
                           [False, False, False], # sample 29: t-a: c1,c2,c3
                           [False, False, False]], # sample 29: i-e: c1,c2,c3
                               
                          [[False, False, False], # sample 30: cit: c1,c2,c3
                           [False, False, False], # sample 30: c-s: c1,c2,c3
                           [False, False, True], # sample 30: t-a: c1,c2,c3
                           [False, False, False]], # sample 30: i-e: c1,c2,c3
                           ])

########################################################################################################################

# Juror 2

j2 = np.array([[[4, 3, 5], # sample 1: cit: c1,c2,c3
                [2, 2, 2], # sample 1: c-s: c1,c2,c3
                [2, 3, 2], # sample 1: t-a: c1,c2,c3
                [4, 3, 4]], # sample 1: i-e: c1,c2,c3

               [[4, 5, 5], # sample 2: cit: c1,c2,c3
                [3, 4, 4], # sample 2: c-s: c1,c2,c3
                [3, 4, 2], # sample 2: t-a: c1,c2,c3
                [3, 3, 3]], # sample 2: i-e: c1,c2,c3
                      
               [[5, 4, 5], # sample 3: cit: c1,c2,c3
                [4, 5, 4], # sample 3: c-s: c1,c2,c3
                [4, 4, 5], # sample 3: t-a: c1,c2,c3
                [4, 4, 5]], # sample 3: i-e: c1,c2,c3

               [[4, 2, 5], # sample 4: cit: c1,c2,c3
                [4, 3, 5], # sample 4: c-s: c1,c2,c3
                [2, 3, 4], # sample 4: t-a: c1,c2,c3
                [2, 2, 4]], # sample 4: i-e: c1,c2,c3
                      
               [[2, 3, 5], # sample 5: cit: c1,c2,c3
                [5, 2, 4], # sample 5: c-s: c1,c2,c3
                [3, 1, 1], # sample 5: t-a: c1,c2,c3
                [4, 4, 3]], # sample 5: i-e: c1,c2,c3

               [[4, 5, 5], # sample 6: cit: c1,c2,c3
                [4, 2, 5], # sample 6: c-s: c1,c2,c3
                [3, 3, 4], # sample 6: t-a: c1,c2,c3
                [3, 3, 3]], # sample 6: i-e: c1,c2,c3
                      
               [[4, 4, 5], # sample 7: cit: c1,c2,c3
                [3, 3, 5], # sample 7: c-s: c1,c2,c3
                [4, 4, 3], # sample 7: t-a: c1,c2,c3
                [3, 3, 4]], # sample 7: i-e: c1,c2,c3

               [[3, 4, 5], # sample 8: cit: c1,c2,c3
                [3, 3, 4], # sample 8: c-s: c1,c2,c3
                [3, 2, 4], # sample 8: t-a: c1,c2,c3
                [4, 4, 3]], # sample 8: i-e: c1,c2,c3
                      
               [[4, 5, 5], # sample 9: cit: c1,c2,c3
                [4, 3, 5], # sample 9: c-s: c1,c2,c3
                [3, 3, 4], # sample 9: t-a: c1,c2,c3
                [4, 3, 4]], # sample 9: i-e: c1,c2,c3

               [[3, 2, 5], # sample 10: cit: c1,c2,c3
                [4, 4, 3], # sample 10: c-s: c1,c2,c3
                [3, 3, 5], # sample 10: t-a: c1,c2,c3
                [4, 4, 2]], # sample 10: i-e: c1,c2,c3
                      
               [[3, 3, 5], # sample 11: cit: c1,c2,c3
                [3, 3, 4], # sample 11: c-s: c1,c2,c3
                [2, 2, 3], # sample 11: t-a: c1,c2,c3
                [4, 3, 1]], # sample 11: i-e: c1,c2,c3

               [[3, 2, 5], # sample 12: cit: c1,c2,c3
                [3, 3, 3], # sample 12: c-s: c1,c2,c3
                [2, 2, 2], # sample 12: t-a: c1,c2,c3
                [3, 4, 1]], # sample 12: i-e: c1,c2,c3
                      
               [[4, 4, 5], # sample 13: cit: c1,c2,c3
                [4, 3, 4], # sample 13: c-s: c1,c2,c3
                [4, 4, 4], # sample 13: t-a: c1,c2,c3
                [3, 3, 3]], # sample 13: i-e: c1,c2,c3

               [[4, 5, 5], # sample 14: cit: c1,c2,c3
                [4, 4, 5], # sample 14: c-s: c1,c2,c3
                [4, 4, 5], # sample 14: t-a: c1,c2,c3
                [3, 2, 2]], # sample 14: i-e: c1,c2,c3
                      
               [[4, 4, 5], # sample 15: cit: c1,c2,c3
                [3, 2, 2], # sample 15: c-s: c1,c2,c3
                [4, 3, 3], # sample 15: t-a: c1,c2,c3
                [4, 4, 2]], # sample 15: i-e: c1,c2,c3

               [[4, 5, 5], # sample 16: cit: c1,c2,c3
                [5, 2, 3], # sample 16: c-s: c1,c2,c3
                [4, 2, 4], # sample 16: t-a: c1,c2,c3
                [2, 3, 1]], # sample 16: i-e: c1,c2,c3
                      
               [[2, 3, 5], # sample 17: cit: c1,c2,c3
                [3, 4, 4], # sample 17: c-s: c1,c2,c3
                [3, 3, 2], # sample 17: t-a: c1,c2,c3
                [4, 3, 4]], # sample 17: i-e: c1,c2,c3

               [[4, 3, 5], # sample 18: cit: c1,c2,c3
                [4, 4, 4], # sample 18: c-s: c1,c2,c3
                [4, 3, 5], # sample 18: t-a: c1,c2,c3
                [4, 3, 3]], # sample 18: i-e: c1,c2,c3
                      
               [[3, 4, 5], # sample 19: cit: c1,c2,c3
                [3, 2, 2], # sample 19: c-s: c1,c2,c3
                [1, 1, 1], # sample 19: t-a: c1,c2,c3
                [3, 4, 4]], # sample 19: i-e: c1,c2,c3

               [[4, 4, 5], # sample 20: cit: c1,c2,c3
                [4, 4, 5], # sample 20: c-s: c1,c2,c3
                [2, 3, 2], # sample 20: t-a: c1,c2,c3
                [3, 3, 4]], # sample 20: i-e: c1,c2,c3
                      
               [[3, 3, 5], # sample 21: cit: c1,c2,c3
                [3, 3, 4], # sample 21: c-s: c1,c2,c3
                [1, 2, 5], # sample 21: t-a: c1,c2,c3
                [3, 3, 1]], # sample 21: i-e: c1,c2,c3

               [[4, 4, 5], # sample 22: cit: c1,c2,c3
                [3, 3, 5], # sample 22: c-s: c1,c2,c3
                [3, 3, 3], # sample 22: t-a: c1,c2,c3
                [2, 3, 4]], # sample 22: i-e: c1,c2,c3
                      
               [[4, 4, 5], # sample 23: cit: c1,c2,c3
                [4, 2, 4], # sample 23: c-s: c1,c2,c3
                [4, 4, 1], # sample 23: t-a: c1,c2,c3
                [1, 5, 1]], # sample 23: i-e: c1,c2,c3

               [[3, 3, 5], # sample 24: cit: c1,c2,c3
                [2, 2, 2], # sample 24: c-s: c1,c2,c3
                [1, 4, 5], # sample 24: t-a: c1,c2,c3
                [4, 4, 2]], # sample 24: i-e: c1,c2,c3
                      
               [[4, 4, 5], # sample 25: cit: c1,c2,c3
                [4, 2, 4], # sample 25: c-s: c1,c2,c3
                [4, 5, 1], # sample 25: t-a: c1,c2,c3
                [3, 5, 3]], # sample 25: i-e: c1,c2,c3

               [[4, 3, 5], # sample 26: cit: c1,c2,c3
                [4, 3, 4], # sample 26: c-s: c1,c2,c3
                [4, 2, 3], # sample 26: t-a: c1,c2,c3
                [4, 3, 3]], # sample 26: i-e: c1,c2,c3                

               [[4, 3, 5], # sample 27: cit: c1,c2,c3
                [4, 4, 4], # sample 27: c-s: c1,c2,c3
                [4, 5, 4], # sample 27: t-a: c1,c2,c3
                [4, 5, 5]], # sample 27: i-e: c1,c2,c3
                      
               [[4, 3, 5], # sample 28: cit: c1,c2,c3,
                [4, 3, 4], # sample 28: c-s: c1,c2,c3
                [2, 3, 4], # sample 28: t-a: c1,c2,c3
                [4, 3, 2]], # sample 28: i-e: c1,c2,c3

               [[4, 4, 5], # sample 29: cit: c1,c2,c3
                [3, 4, 4], # sample 29: c-s: c1,c2,c3
                [4, 2, 4], # sample 29: t-a: c1,c2,c3
                [3, 3, 4]], # sample 29: i-e: c1,c2,c3
                      
               [[3, 3, 5], # sample 30: cit: c1,c2,c3
                [2, 2, 3], # sample 30: c-s: c1,c2,c3
                [2, 2, 2], # sample 30: t-a: c1,c2,c3
                [3, 3, 2]] # sample 30: i-e: c1,c2,c3
            ])


j2_confidence = np.array([[[False, False, False], # sample 1: cit: c1,c2,c3
                           [False, False, False], # sample 1: c-s: c1,c2,c3
                           [False, False, False], # sample 1: t-a: c1,c2,c3
                           [False, False, False]], # sample 1: i-e: c1,c2,c3

                          [[False, False, False], # sample 2: cit: c1,c2,c3
                           [False, False, False], # sample 2: c-s: c1,c2,c3
                           [False, False, False], # sample 2: t-a: c1,c2,c3
                           [False, False, False]], # sample 2: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 3: cit: c1,c2,c3
                           [False, False, False], # sample 3: c-s: c1,c2,c3
                           [False, False, False], # sample 3: t-a: c1,c2,c3
                           [False, False, False]], # sample 3: i-e: c1,c2,c3

                          [[False, False, False], # sample 4: cit: c1,c2,c3
                           [False, False, False], # sample 4: c-s: c1,c2,c3
                           [False, False, False], # sample 4: t-a: c1,c2,c3
                           [False, False, False]], # sample 4: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 5: cit: c1,c2,c3
                           [False, False, False], # sample 5: c-s: c1,c2,c3
                           [False, False, False], # sample 5: t-a: c1,c2,c3
                           [False, False, False]], # sample 5: i-e: c1,c2,c3

                          [[False, False, False], # sample 6: cit: c1,c2,c3
                           [False, False, False], # sample 6: c-s: c1,c2,c3
                           [False, False, False], # sample 6: t-a: c1,c2,c3
                           [False, False, False]], # sample 6: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 7: cit: c1,c2,c3
                           [False, False, False], # sample 7: c-s: c1,c2,c3
                           [False, False, False], # sample 7: t-a: c1,c2,c3
                           [False, False, False]], # sample 7: i-e: c1,c2,c3

                          [[False, False, False], # sample 8: cit: c1,c2,c3
                           [False, False, False], # sample 8: c-s: c1,c2,c3
                           [False, False, False], # sample 8: t-a: c1,c2,c3
                           [False, False, False]], # sample 8: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 9: cit: c1,c2,c3
                           [False, False, False], # sample 9: c-s: c1,c2,c3
                           [False, False, False], # sample 9: t-a: c1,c2,c3
                           [False, False, False]], # sample 9: i-e: c1,c2,c3

                          [[False, False, False], # sample 10: cit: c1,c2,c3
                           [False, False, False], # sample 10: c-s: c1,c2,c3
                           [False, False, False], # sample 10: t-a: c1,c2,c3
                           [False, False, False]], # sample 10: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 11: cit: c1,c2,c3
                           [False, False, False], # sample 11: c-s: c1,c2,c3
                           [False, False, False], # sample 11: t-a: c1,c2,c3
                           [False, False, True]], # sample 11: i-e: c1,c2,c3

                          [[False, False, False], # sample 12: cit: c1,c2,c3
                           [False, False, False], # sample 12: c-s: c1,c2,c3
                           [False, False, False], # sample 12: t-a: c1,c2,c3
                           [False, False, False]], # sample 12: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 13: cit: c1,c2,c3
                           [False, False, False], # sample 13: c-s: c1,c2,c3
                           [False, False, False], # sample 13: t-a: c1,c2,c3
                           [False, False, False]], # sample 13: i-e: c1,c2,c3

                          [[False, False, False], # sample 14: cit: c1,c2,c3
                           [False, False, False], # sample 14: c-s: c1,c2,c3
                           [False, False, False], # sample 14: t-a: c1,c2,c3
                           [False, False, False]], # sample 14: i-e: c1,c2,c3
                               
                          [[False, False, False], # sample 15: cit: c1,c2,c3
                           [False, False, False], # sample 15: c-s: c1,c2,c3
                           [False, False, False], # sample 15: t-a: c1,c2,c3
                           [False, False, False]], # sample 15: i-e: c1,c2,c3

                          [[False, False, False], # sample 16: cit: c1,c2,c3
                           [False, False, False], # sample 16: c-s: c1,c2,c3
                           [False, False, False], # sample 16: t-a: c1,c2,c3
                           [False, False, False]], # sample 16: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 17: cit: c1,c2,c3
                           [False, False, False], # sample 17: c-s: c1,c2,c3
                           [False, False, False], # sample 17: t-a: c1,c2,c3
                           [False, False, False]], # sample 17: i-e: c1,c2,c3

                          [[False, False, False], # sample 18: cit: c1,c2,c3
                           [False, False, False], # sample 18: c-s: c1,c2,c3
                           [False, False, False], # sample 18: t-a: c1,c2,c3
                           [False, False, False]], # sample 18: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 19: cit: c1,c2,c3
                           [False, False, False], # sample 19: c-s: c1,c2,c3
                           [True, True, False], # sample 19: t-a: c1,c2,c3
                           [False, False, False]], # sample 19: i-e: c1,c2,c3

                          [[False, False, False], # sample 20: cit: c1,c2,c3
                           [False, False, False], # sample 20: c-s: c1,c2,c3
                           [False, False, False], # sample 20: t-a: c1,c2,c3
                           [False, False, False]], # sample 20: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 21: cit: c1,c2,c3
                           [False, False, False], # sample 21: c-s: c1,c2,c3
                           [False, False, False], # sample 21: t-a: c1,c2,c3
                           [False, False, False]], # sample 21: i-e: c1,c2,c3

                          [[False, False, False], # sample 22: cit: c1,c2,c3
                           [False, False, False], # sample 22: c-s: c1,c2,c3
                           [False, False, False], # sample 22: t-a: c1,c2,c3
                           [False, False, False]], # sample 22: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 23: cit: c1,c2,c3
                           [False, False, False], # sample 23: c-s: c1,c2,c3
                           [False, False, False], # sample 23: t-a: c1,c2,c3
                           [True, False, True]], # sample 23: i-e: c1,c2,c3

                          [[False, False, False], # sample 24: cit: c1,c2,c3
                           [False, False, False], # sample 24: c-s: c1,c2,c3
                           [False, False, False], # sample 24: t-a: c1,c2,c3
                           [False, False, False]], # sample 24: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 25: cit: c1,c2,c3
                           [False, False, False], # sample 25: c-s: c1,c2,c3
                           [False, False, False], # sample 25: t-a: c1,c2,c3
                           [False, False, False]], # sample 25: i-e: c1,c2,c3

                          [[False, False, False], # sample 26: cit: c1,c2,c3
                           [False, False, False], # sample 26: c-s: c1,c2,c3
                           [False, False, False], # sample 26: t-a: c1,c2,c3
                           [False, False, False]], # sample 26: i-e: c1,c2,c3                

                          [[False, False, False], # sample 27: cit: c1,c2,c3
                           [False, False, False], # sample 27: c-s: c1,c2,c3
                           [False, False, False], # sample 27: t-a: c1,c2,c3
                           [False, False, False]], # sample 27: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 28: cit: c1,c2,c3,
                           [False, False, False], # sample 28: c-s: c1,c2,c3
                           [False, False, False], # sample 28: t-a: c1,c2,c3
                           [False, False, False]], # sample 28: i-e: c1,c2,c3

                          [[False, False, False], # sample 29: cit: c1,c2,c3
                           [False, False, False], # sample 29: c-s: c1,c2,c3
                           [False, False, False], # sample 29: t-a: c1,c2,c3
                           [False, False, False]], # sample 29: i-e: c1,c2,c3
                               
                          [[False, False, False], # sample 30: cit: c1,c2,c3
                           [False, False, False], # sample 30: c-s: c1,c2,c3
                           [False, False, False], # sample 30: t-a: c1,c2,c3
                           [False, False, False]], # sample 30: i-e: c1,c2,c3
                           ])


########################################################################################################################

# Juror 3

j3 = np.array([[[4, 4, 5], # sample 1: cit: c1,c2,c3
                [2, 3, 4], # sample 1: c-s: c1,c2,c3
                [2, 2, 1], # sample 1: t-a: c1,c2,c3
                [4, 3, 3]], # sample 1: i-e: c1,c2,c3

               [[5, 2, 5], # sample 2: cit: c1,c2,c3
                [4, 5, 3], # sample 2: c-s: c1,c2,c3
                [5, 4, 2], # sample 2: t-a: c1,c2,c3
                [1, 4, 1]], # sample 2: i-e: c1,c2,c3
                      
               [[5, 3, 5], # sample 3: cit: c1,c2,c3
                [4, 4, 1], # sample 3: c-s: c1,c2,c3
                [4, 5, 2], # sample 3: t-a: c1,c2,c3
                [5, 4, 5]], # sample 3: i-e: c1,c2,c3

               [[2, 4, 5], # sample 4: cit: c1,c2,c3
                [3, 4, 1], # sample 4: c-s: c1,c2,c3
                [2, 3, 2], # sample 4: t-a: c1,c2,c3
                [2, 1, 1]], # sample 4: i-e: c1,c2,c3
                      
               [[4, 4, 5], # sample 5: cit: c1,c2,c3
                [1, 1, 4], # sample 5: c-s: c1,c2,c3
                [1, 1, 1], # sample 5: t-a: c1,c2,c3
                [3, 5, 2]], # sample 5: i-e: c1,c2,c3

               [[4, 4, 5], # sample 6: cit: c1,c2,c3
                [2, 3, 1], # sample 6: c-s: c1,c2,c3
                [2, 2, 1], # sample 6: t-a: c1,c2,c3
                [4, 2, 3]], # sample 6: i-e: c1,c2,c3
                      
               [[5, 4, 5], # sample 7: cit: c1,c2,c3
                [2, 3, 4], # sample 7: c-s: c1,c2,c3
                [1, 2, 3], # sample 7: t-a: c1,c2,c3
                [1, 4, 1]], # sample 7: i-e: c1,c2,c3

               [[3, 4, 5], # sample 8: cit: c1,c2,c3
                [3, 2, 2], # sample 8: c-s: c1,c2,c3
                [3, 2, 2], # sample 8: t-a: c1,c2,c3
                [1, 4, 1]], # sample 8: i-e: c1,c2,c3
                      
               [[3, 4, 5], # sample 9: cit: c1,c2,c3
                [2, 5, 4], # sample 9: c-s: c1,c2,c3
                [2, 3, 2], # sample 9: t-a: c1,c2,c3
                [1, 1, 1]], # sample 9: i-e: c1,c2,c3

               [[4, 5, 5], # sample 10: cit: c1,c2,c3
                [2, 5, 2], # sample 10: c-s: c1,c2,c3
                [4, 3, 4], # sample 10: t-a: c1,c2,c3
                [1, 2, 2]], # sample 10: i-e: c1,c2,c3
                      
               [[4, 3, 5], # sample 11: cit: c1,c2,c3
                [4, 2, 5], # sample 11: c-s: c1,c2,c3
                [1, 2, 1], # sample 11: t-a: c1,c2,c3
                [1, 4, 1]], # sample 11: i-e: c1,c2,c3

               [[4, 2, 5], # sample 12: cit: c1,c2,c3
                [4, 2, 1], # sample 12: c-s: c1,c2,c3
                [1, 1, 1], # sample 12: t-a: c1,c2,c3
                [1, 3, 1]], # sample 12: i-e: c1,c2,c3
                      
               [[4, 3, 5], # sample 13: cit: c1,c2,c3
                [3, 4, 1], # sample 13: c-s: c1,c2,c3
                [2, 4, 2], # sample 13: t-a: c1,c2,c3
                [2, 3, 2]], # sample 13: i-e: c1,c2,c3

               [[1, 1, 5], # sample 14: cit: c1,c2,c3
                [1, 2, 1], # sample 14: c-s: c1,c2,c3
                [1, 3, 2], # sample 14: t-a: c1,c2,c3
                [1, 3, 1]], # sample 14: i-e: c1,c2,c3
                      
               [[5, 5, 5], # sample 15: cit: c1,c2,c3
                [2, 4, 1], # sample 15: c-s: c1,c2,c3
                [2, 1, 1], # sample 15: t-a: c1,c2,c3
                [2, 4, 1]], # sample 15: i-e: c1,c2,c3

               [[4, 5, 5], # sample 16: cit: c1,c2,c3
                [2, 2, 2], # sample 16: c-s: c1,c2,c3
                [2, 1, 2], # sample 16: t-a: c1,c2,c3
                [3, 2, 2]], # sample 16: i-e: c1,c2,c3
                      
               [[4, 2, 5], # sample 17: cit: c1,c2,c3
                [4, 5, 1], # sample 17: c-s: c1,c2,c3
                [2, 2, 1], # sample 17: t-a: c1,c2,c3
                [4, 3, 1]], # sample 17: i-e: c1,c2,c3

               [[3, 2, 5], # sample 18: cit: c1,c2,c3
                [5, 2, 4], # sample 18: c-s: c1,c2,c3
                [3, 4, 1], # sample 18: t-a: c1,c2,c3
                [3, 5, 2]], # sample 18: i-e: c1,c2,c3
                      
               [[4, 4, 5], # sample 19: cit: c1,c2,c3
                [1, 1, 1], # sample 19: c-s: c1,c2,c3
                [3, 2, 1], # sample 19: t-a: c1,c2,c3
                [1, 2, 1]], # sample 19: i-e: c1,c2,c3

               [[5, 4, 5], # sample 20: cit: c1,c2,c3
                [3, 3, 5], # sample 20: c-s: c1,c2,c3
                [4, 4, 2], # sample 20: t-a: c1,c2,c3
                [2, 3, 2]], # sample 20: i-e: c1,c2,c3
                      
               [[5, 5, 5], # sample 21: cit: c1,c2,c3
                [3, 2, 4], # sample 21: c-s: c1,c2,c3
                [2, 1, 4], # sample 21: t-a: c1,c2,c3
                [1, 4, 1]], # sample 21: i-e: c1,c2,c3

               [[3, 4, 5], # sample 22: cit: c1,c2,c3
                [3, 4, 1], # sample 22: c-s: c1,c2,c3
                [3, 4, 2], # sample 22: t-a: c1,c2,c3
                [4, 4, 2]], # sample 22: i-e: c1,c2,c3
                      
               [[2, 3, 5], # sample 23: cit: c1,c2,c3
                [2, 1, 5], # sample 23: c-s: c1,c2,c3
                [1, 3, 1], # sample 23: t-a: c1,c2,c3
                [1, 4, 2]], # sample 23: i-e: c1,c2,c3

               [[4, 4, 5], # sample 24: cit: c1,c2,c3
                [3, 3, 1], # sample 24: c-s: c1,c2,c3
                [5, 5, 5], # sample 24: t-a: c1,c2,c3
                [1, 4, 1]], # sample 24: i-e: c1,c2,c3
                      
               [[5, 5, 5], # sample 25: cit: c1,c2,c3
                [3, 2, 4], # sample 25: c-s: c1,c2,c3
                [1, 4, 2], # sample 25: t-a: c1,c2,c3
                [2, 4, 2]], # sample 25: i-e: c1,c2,c3

               [[5, 5, 5], # sample 26: cit: c1,c2,c3
                [4, 4, 3], # sample 26: c-s: c1,c2,c3
                [4, 1, 5], # sample 26: t-a: c1,c2,c3
                [4, 2, 4]], # sample 26: i-e: c1,c2,c3                

               [[4, 3, 5], # sample 27: cit: c1,c2,c3
                [5, 4, 4], # sample 27: c-s: c1,c2,c3
                [4, 3, 1], # sample 27: t-a: c1,c2,c3
                [1, 3, 1]], # sample 27: i-e: c1,c2,c3
                      
               [[4, 2, 5], # sample 28: cit: c1,c2,c3,
                [4, 5, 2], # sample 28: c-s: c1,c2,c3
                [4, 4, 2], # sample 28: t-a: c1,c2,c3
                [4, 5, 5]], # sample 28: i-e: c1,c2,c3

               [[4, 4, 5], # sample 29: cit: c1,c2,c3
                [4, 4, 4], # sample 29: c-s: c1,c2,c3
                [2, 1, 1], # sample 29: t-a: c1,c2,c3
                [3, 2, 3]], # sample 29: i-e: c1,c2,c3
                      
               [[5, 3, 5], # sample 30: cit: c1,c2,c3
                [4, 2, 2], # sample 30: c-s: c1,c2,c3
                [2, 4, 2], # sample 30: t-a: c1,c2,c3
                [4, 3, 5]], # sample 30: i-e: c1,c2,c3
            ])


j3_confidence = np.array([[[False, False, False], # sample 1: cit: c1,c2,c3
                           [False, False, False], # sample 1: c-s: c1,c2,c3
                           [False, False, False], # sample 1: t-a: c1,c2,c3
                           [False, False, False]], # sample 1: i-e: c1,c2,c3

                          [[False, False, False], # sample 2: cit: c1,c2,c3
                           [False, False, False], # sample 2: c-s: c1,c2,c3
                           [False, False, True], # sample 2: t-a: c1,c2,c3
                           [False, False, False]], # sample 2: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 3: cit: c1,c2,c3
                           [False, False, False], # sample 3: c-s: c1,c2,c3
                           [False, False, False], # sample 3: t-a: c1,c2,c3
                           [False, False, False]], # sample 3: i-e: c1,c2,c3

                          [[False, False, False], # sample 4: cit: c1,c2,c3
                           [False, False, True], # sample 4: c-s: c1,c2,c3
                           [False, False, True], # sample 4: t-a: c1,c2,c3
                           [False, False, True]], # sample 4: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 5: cit: c1,c2,c3
                           [False, False, False], # sample 5: c-s: c1,c2,c3
                           [False, False, False], # sample 5: t-a: c1,c2,c3
                           [False, False, False]], # sample 5: i-e: c1,c2,c3

                          [[False, False, False], # sample 6: cit: c1,c2,c3
                           [False, False, True], # sample 6: c-s: c1,c2,c3
                           [False, False, True], # sample 6: t-a: c1,c2,c3
                           [False, False, False]], # sample 6: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 7: cit: c1,c2,c3
                           [False, False, False], # sample 7: c-s: c1,c2,c3
                           [False, False, False], # sample 7: t-a: c1,c2,c3
                           [False, False, False]], # sample 7: i-e: c1,c2,c3

                          [[False, False, False], # sample 8: cit: c1,c2,c3
                           [False, False, False], # sample 8: c-s: c1,c2,c3
                           [False, False, False], # sample 8: t-a: c1,c2,c3
                           [False, False, False]], # sample 8: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 9: cit: c1,c2,c3
                           [False, False, False], # sample 9: c-s: c1,c2,c3
                           [False, False, False], # sample 9: t-a: c1,c2,c3
                           [False, False, False]], # sample 9: i-e: c1,c2,c3

                          [[False, False, False], # sample 10: cit: c1,c2,c3
                           [False, False, False], # sample 10: c-s: c1,c2,c3
                           [False, False, False], # sample 10: t-a: c1,c2,c3
                           [False, False, False]], # sample 10: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 11: cit: c1,c2,c3
                           [False, False, False], # sample 11: c-s: c1,c2,c3
                           [False, False, False], # sample 11: t-a: c1,c2,c3
                           [False, False, False]], # sample 11: i-e: c1,c2,c3

                          [[False, False, False], # sample 12: cit: c1,c2,c3
                           [False, False, False], # sample 12: c-s: c1,c2,c3
                           [False, False, True], # sample 12: t-a: c1,c2,c3
                           [False, False, False]], # sample 12: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 13: cit: c1,c2,c3
                           [False, False, True], # sample 13: c-s: c1,c2,c3
                           [False, False, False], # sample 13: t-a: c1,c2,c3
                           [False, False, False]], # sample 13: i-e: c1,c2,c3

                          [[False, False, False], # sample 14: cit: c1,c2,c3
                           [False, False, True], # sample 14: c-s: c1,c2,c3
                           [False, False, True], # sample 14: t-a: c1,c2,c3
                           [False, False, False]], # sample 14: i-e: c1,c2,c3
                               
                          [[False, False, False], # sample 15: cit: c1,c2,c3
                           [False, False, False], # sample 15: c-s: c1,c2,c3
                           [False, False, False], # sample 15: t-a: c1,c2,c3
                           [False, False, False]], # sample 15: i-e: c1,c2,c3

                          [[False, False, False], # sample 16: cit: c1,c2,c3
                           [False, False, True], # sample 16: c-s: c1,c2,c3
                           [False, False, True], # sample 16: t-a: c1,c2,c3
                           [False, False, False]], # sample 16: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 17: cit: c1,c2,c3
                           [False, False, True], # sample 17: c-s: c1,c2,c3
                           [False, False, True], # sample 17: t-a: c1,c2,c3
                           [False, False, True]], # sample 17: i-e: c1,c2,c3

                          [[False, False, False], # sample 18: cit: c1,c2,c3
                           [False, False, False], # sample 18: c-s: c1,c2,c3
                           [False, False, False], # sample 18: t-a: c1,c2,c3
                           [False, False, False]], # sample 18: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 19: cit: c1,c2,c3
                           [False, False, False], # sample 19: c-s: c1,c2,c3
                           [False, False, True], # sample 19: t-a: c1,c2,c3
                           [False, False, False]], # sample 19: i-e: c1,c2,c3

                          [[False, False, False], # sample 20: cit: c1,c2,c3
                           [False, False, False], # sample 20: c-s: c1,c2,c3
                           [False, False, True], # sample 20: t-a: c1,c2,c3
                           [False, False, False]], # sample 20: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 21: cit: c1,c2,c3
                           [False, False, False], # sample 21: c-s: c1,c2,c3
                           [False, False, False], # sample 21: t-a: c1,c2,c3
                           [False, False, False]], # sample 21: i-e: c1,c2,c3

                          [[False, False, False], # sample 22: cit: c1,c2,c3
                           [False, False, True], # sample 22: c-s: c1,c2,c3
                           [False, False, True], # sample 22: t-a: c1,c2,c3
                           [False, False, False]], # sample 22: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 23: cit: c1,c2,c3
                           [False, False, False], # sample 23: c-s: c1,c2,c3
                           [False, False, True], # sample 23: t-a: c1,c2,c3
                           [False, False, False]], # sample 23: i-e: c1,c2,c3

                          [[False, False, False], # sample 24: cit: c1,c2,c3
                           [False, False, True], # sample 24: c-s: c1,c2,c3
                           [False, False, False], # sample 24: t-a: c1,c2,c3
                           [False, False, False]], # sample 24: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 25: cit: c1,c2,c3
                           [False, False, False], # sample 25: c-s: c1,c2,c3
                           [False, False, True], # sample 25: t-a: c1,c2,c3
                           [False, False, False]], # sample 25: i-e: c1,c2,c3

                          [[False, False, False], # sample 26: cit: c1,c2,c3
                           [False, False, False], # sample 26: c-s: c1,c2,c3
                           [False, False, False], # sample 26: t-a: c1,c2,c3
                           [False, False, False]], # sample 26: i-e: c1,c2,c3                

                          [[False, False, False], # sample 27: cit: c1,c2,c3
                           [False, False, False], # sample 27: c-s: c1,c2,c3
                           [False, False, True], # sample 27: t-a: c1,c2,c3
                           [False, False, False]], # sample 27: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 28: cit: c1,c2,c3,
                           [False, False, True], # sample 28: c-s: c1,c2,c3
                           [False, False, True], # sample 28: t-a: c1,c2,c3
                           [False, False, False]], # sample 28: i-e: c1,c2,c3

                          [[False, False, False], # sample 29: cit: c1,c2,c3
                           [False, False, False], # sample 29: c-s: c1,c2,c3
                           [False, False, True], # sample 29: t-a: c1,c2,c3
                           [False, False, False]], # sample 29: i-e: c1,c2,c3
                               
                          [[False, False, False], # sample 30: cit: c1,c2,c3
                           [False, False, True], # sample 30: c-s: c1,c2,c3
                           [False, False, True], # sample 30: t-a: c1,c2,c3
                           [False, False, False]], # sample 30: i-e: c1,c2,c3
                           ])

########################################################################################################################

# Juror 4

j4 = np.array([[[3, 4, 5], # sample 1: cit: c1,c2,c3
                [5, 3, 5], # sample 1: c-s: c1,c2,c3
                [4, 2, 3], # sample 1: t-a: c1,c2,c3
                [4, 1, 3]], # sample 1: i-e: c1,c2,c3

               [[4, 3, 5], # sample 2: cit: c1,c2,c3
                [5, 5, 2], # sample 2: c-s: c1,c2,c3
                [5, 3, 3], # sample 2: t-a: c1,c2,c3
                [1, 1, 1]], # sample 2: i-e: c1,c2,c3
                      
               [[4, 3, 5], # sample 3: cit: c1,c2,c3
                [4, 3, 4], # sample 3: c-s: c1,c2,c3
                [5, 5, 3], # sample 3: t-a: c1,c2,c3
                [5, 5, 5]], # sample 3: i-e: c1,c2,c3

               [[4, 5, 5], # sample 4: cit: c1,c2,c3
                [5, 2, 2], # sample 4: c-s: c1,c2,c3
                [2, 3, 4], # sample 4: t-a: c1,c2,c3
                [4, 4, 1]], # sample 4: i-e: c1,c2,c3
                      
               [[5, 4, 5], # sample 5: cit: c1,c2,c3
                [3, 2, 3], # sample 5: c-s: c1,c2,c3
                [4, 4, 2], # sample 5: t-a: c1,c2,c3
                [4, 2, 2]], # sample 5: i-e: c1,c2,c3

               [[5, 5, 5], # sample 6: cit: c1,c2,c3
                [5, 3, 4], # sample 6: c-s: c1,c2,c3
                [4, 3, 4], # sample 6: t-a: c1,c2,c3
                [5, 3, 4]], # sample 6: i-e: c1,c2,c3
                      
               [[5, 4, 5], # sample 7: cit: c1,c2,c3
                [5, 3, 5], # sample 7: c-s: c1,c2,c3
                [5, 4, 3], # sample 7: t-a: c1,c2,c3
                [2, 3, 1]], # sample 7: i-e: c1,c2,c3

               [[5, 4, 5], # sample 8: cit: c1,c2,c3
                [5, 5, 2], # sample 8: c-s: c1,c2,c3
                [5, 5, 3], # sample 8: t-a: c1,c2,c3
                [5, 4, 1]], # sample 8: i-e: c1,c2,c3
                      
               [[4, 5, 5], # sample 9: cit: c1,c2,c3
                [3, 5, 3], # sample 9: c-s: c1,c2,c3
                [3, 4, 1], # sample 9: t-a: c1,c2,c3
                [4, 3, 2]], # sample 9: i-e: c1,c2,c3

               [[3, 2, 5], # sample 10: cit: c1,c2,c3
                [4, 4, 1], # sample 10: c-s: c1,c2,c3
                [5, 3, 3], # sample 10: t-a: c1,c2,c3
                [1, 5, 1]], # sample 10: i-e: c1,c2,c3
                      
               [[2, 3, 5], # sample 11: cit: c1,c2,c3
                [3, 1, 3], # sample 11: c-s: c1,c2,c3
                [2, 1, 2], # sample 11: t-a: c1,c2,c3
                [1, 4, 1]], # sample 11: i-e: c1,c2,c3

               [[3, 1, 5], # sample 12: cit: c1,c2,c3
                [3, 2, 3], # sample 12: c-s: c1,c2,c3
                [5, 3, 3], # sample 12: t-a: c1,c2,c3
                [5, 4, 1]], # sample 12: i-e: c1,c2,c3
                      
               [[5, 5, 5], # sample 13: cit: c1,c2,c3
                [3, 4, 4], # sample 13: c-s: c1,c2,c3
                [4, 5, 4], # sample 13: t-a: c1,c2,c3
                [5, 5, 4]], # sample 13: i-e: c1,c2,c3

               [[5, 4, 5], # sample 14: cit: c1,c2,c3
                [5, 4, 4], # sample 14: c-s: c1,c2,c3
                [4, 4, 4], # sample 14: t-a: c1,c2,c3
                [4, 3, 1]], # sample 14: i-e: c1,c2,c3
                      
               [[4, 4, 5], # sample 15: cit: c1,c2,c3
                [5, 4, 3], # sample 15: c-s: c1,c2,c3
                [5, 4, 1], # sample 15: t-a: c1,c2,c3
                [3, 3, 1]], # sample 15: i-e: c1,c2,c3

               [[5, 5, 5], # sample 16: cit: c1,c2,c3
                [4, 3, 2], # sample 16: c-s: c1,c2,c3
                [4, 2, 2], # sample 16: t-a: c1,c2,c3
                [5, 5, 3]], # sample 16: i-e: c1,c2,c3
                      
               [[4, 4, 5], # sample 17: cit: c1,c2,c3
                [3, 4, 4], # sample 17: c-s: c1,c2,c3
                [4, 5, 2], # sample 17: t-a: c1,c2,c3
                [5, 4, 4]], # sample 17: i-e: c1,c2,c3

               [[4, 4, 5], # sample 18: cit: c1,c2,c3
                [5, 4, 4], # sample 18: c-s: c1,c2,c3
                [4, 5, 3], # sample 18: t-a: c1,c2,c3
                [5, 5, 4]], # sample 18: i-e: c1,c2,c3
                      
               [[5, 5, 5], # sample 19: cit: c1,c2,c3
                [3, 3, 4], # sample 19: c-s: c1,c2,c3
                [2, 2, 1], # sample 19: t-a: c1,c2,c3
                [3, 4, 3]], # sample 19: i-e: c1,c2,c3

               [[3, 4, 5], # sample 20: cit: c1,c2,c3
                [4, 4, 4], # sample 20: c-s: c1,c2,c3
                [2, 5, 2], # sample 20: t-a: c1,c2,c3
                [3, 4, 4]], # sample 20: i-e: c1,c2,c3
                      
               [[4, 4, 5], # sample 21: cit: c1,c2,c3
                [2, 4, 3], # sample 21: c-s: c1,c2,c3
                [1, 1, 4], # sample 21: t-a: c1,c2,c3
                [5, 5, 1]], # sample 21: i-e: c1,c2,c3

               [[4, 5, 5], # sample 22: cit: c1,c2,c3
                [5, 3, 5], # sample 22: c-s: c1,c2,c3
                [5, 5, 2], # sample 22: t-a: c1,c2,c3
                [2, 4, 4]], # sample 22: i-e: c1,c2,c3
                      
               [[4, 5, 5], # sample 23: cit: c1,c2,c3
                [4, 2, 5], # sample 23: c-s: c1,c2,c3
                [3, 2, 1], # sample 23: t-a: c1,c2,c3
                [1, 3, 1]], # sample 23: i-e: c1,c2,c3

               [[5, 5, 5], # sample 24: cit: c1,c2,c3
                [4, 3, 3], # sample 24: c-s: c1,c2,c3
                [4, 5, 2], # sample 24: t-a: c1,c2,c3
                [4, 4, 4]], # sample 24: i-e: c1,c2,c3
                      
               [[4, 4, 5], # sample 25: cit: c1,c2,c3
                [4, 4, 4], # sample 25: c-s: c1,c2,c3
                [4, 4, 3], # sample 25: t-a: c1,c2,c3
                [3, 3, 1]], # sample 25: i-e: c1,c2,c3

               [[5, 5, 5], # sample 26: cit: c1,c2,c3
                [4, 4, 4], # sample 26: c-s: c1,c2,c3
                [4, 3, 4], # sample 26: t-a: c1,c2,c3
                [2, 4, 4]], # sample 26: i-e: c1,c2,c3                

               [[5, 4, 5], # sample 27: cit: c1,c2,c3
                [4, 4, 4], # sample 27: c-s: c1,c2,c3
                [5, 5, 2], # sample 27: t-a: c1,c2,c3
                [4, 3, 3]], # sample 27: i-e: c1,c2,c3
                      
               [[5, 4, 5], # sample 28: cit: c1,c2,c3
                [5, 5, 4], # sample 28: c-s: c1,c2,c3
                [3, 4, 3], # sample 28: t-a: c1,c2,c3
                [5, 5, 2]], # sample 28: i-e: c1,c2,c3

               [[5, 5, 5], # sample 29: cit: c1,c2,c3
                [4, 4, 5], # sample 29: c-s: c1,c2,c3
                [5, 4, 5], # sample 29: t-a: c1,c2,c3
                [5, 5, 1]], # sample 29: i-e: c1,c2,c3
                      
               [[5, 5, 5], # sample 30: cit: c1,c2,c3
                [3, 3, 2], # sample 30: c-s: c1,c2,c3
                [4, 4, 2], # sample 30: t-a: c1,c2,c3
                [4, 4, 3]], # sample 30: i-e: c1,c2,c3
            ])


j4_confidence = np.array([[[False, False, False], # sample 1: cit: c1,c2,c3
                           [False, False, False], # sample 1: c-s: c1,c2,c3
                           [False, False, False], # sample 1: t-a: c1,c2,c3
                           [False, False, True]], # sample 1: i-e: c1,c2,c3

                          [[False, False, False], # sample 2: cit: c1,c2,c3
                           [False, False, True], # sample 2: c-s: c1,c2,c3
                           [False, False, True], # sample 2: t-a: c1,c2,c3
                           [False, False, True]], # sample 2: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 3: cit: c1,c2,c3
                           [False, False, True], # sample 3: c-s: c1,c2,c3
                           [False, False, True], # sample 3: t-a: c1,c2,c3
                           [False, False, True]], # sample 3: i-e: c1,c2,c3

                          [[False, False, False], # sample 4: cit: c1,c2,c3
                           [False, False, False], # sample 4: c-s: c1,c2,c3
                           [False, False, False], # sample 4: t-a: c1,c2,c3
                           [False, False, False]], # sample 4: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 5: cit: c1,c2,c3
                           [False, False, False], # sample 5: c-s: c1,c2,c3
                           [False, False, True], # sample 5: t-a: c1,c2,c3
                           [False, False, False]], # sample 5: i-e: c1,c2,c3

                          [[False, False, False], # sample 6: cit: c1,c2,c3
                           [False, False, False], # sample 6: c-s: c1,c2,c3
                           [False, False, False], # sample 6: t-a: c1,c2,c3
                           [False, False, False]], # sample 6: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 7: cit: c1,c2,c3
                           [False, False, False], # sample 7: c-s: c1,c2,c3
                           [False, False, False], # sample 7: t-a: c1,c2,c3
                           [False, False, True]], # sample 7: i-e: c1,c2,c3

                          [[False, False, False], # sample 8: cit: c1,c2,c3
                           [False, False, True], # sample 8: c-s: c1,c2,c3
                           [False, False, True], # sample 8: t-a: c1,c2,c3
                           [False, False, True]], # sample 8: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 9: cit: c1,c2,c3
                           [False, False, False], # sample 9: c-s: c1,c2,c3
                           [False, False, False], # sample 9: t-a: c1,c2,c3
                           [False, False, True]], # sample 9: i-e: c1,c2,c3

                          [[False, False, False], # sample 10: cit: c1,c2,c3
                           [False, False, True], # sample 10: c-s: c1,c2,c3
                           [False, False, False], # sample 10: t-a: c1,c2,c3
                           [False, False, False]], # sample 10: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 11: cit: c1,c2,c3
                           [False, False, False], # sample 11: c-s: c1,c2,c3
                           [False, False, True], # sample 11: t-a: c1,c2,c3
                           [False, False, True]], # sample 11: i-e: c1,c2,c3

                          [[False, False, False], # sample 12: cit: c1,c2,c3
                           [False, False, True], # sample 12: c-s: c1,c2,c3
                           [False, False, False], # sample 12: t-a: c1,c2,c3
                           [False, False, True]], # sample 12: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 13: cit: c1,c2,c3
                           [False, False, False], # sample 13: c-s: c1,c2,c3
                           [False, False, False], # sample 13: t-a: c1,c2,c3
                           [False, False, False]], # sample 13: i-e: c1,c2,c3

                          [[False, False, False], # sample 14: cit: c1,c2,c3
                           [False, False, False], # sample 14: c-s: c1,c2,c3
                           [False, False, False], # sample 14: t-a: c1,c2,c3
                           [False, False, True]], # sample 14: i-e: c1,c2,c3
                               
                          [[False, False, False], # sample 15: cit: c1,c2,c3
                           [False, False, False], # sample 15: c-s: c1,c2,c3
                           [False, False, True], # sample 15: t-a: c1,c2,c3
                           [False, False, True]], # sample 15: i-e: c1,c2,c3

                          [[False, False, False], # sample 16: cit: c1,c2,c3
                           [False, False, False], # sample 16: c-s: c1,c2,c3
                           [False, False, False], # sample 16: t-a: c1,c2,c3
                           [False, False, True]], # sample 16: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 17: cit: c1,c2,c3
                           [False, False, False], # sample 17: c-s: c1,c2,c3
                           [False, False, True], # sample 17: t-a: c1,c2,c3
                           [False, False, False]], # sample 17: i-e: c1,c2,c3

                          [[False, False, False], # sample 18: cit: c1,c2,c3
                           [False, False, False], # sample 18: c-s: c1,c2,c3
                           [False, False, False], # sample 18: t-a: c1,c2,c3
                           [False, False, False]], # sample 18: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 19: cit: c1,c2,c3
                           [False, False, False], # sample 19: c-s: c1,c2,c3
                           [False, False, True], # sample 19: t-a: c1,c2,c3
                           [False, False, True]], # sample 19: i-e: c1,c2,c3

                          [[False, False, False], # sample 20: cit: c1,c2,c3
                           [False, False, False], # sample 20: c-s: c1,c2,c3
                           [False, False, True], # sample 20: t-a: c1,c2,c3
                           [False, False, True]], # sample 20: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 21: cit: c1,c2,c3
                           [False, False, True], # sample 21: c-s: c1,c2,c3
                           [False, False, True], # sample 21: t-a: c1,c2,c3
                           [False, False, True]], # sample 21: i-e: c1,c2,c3

                          [[False, False, False], # sample 22: cit: c1,c2,c3
                           [False, False, False], # sample 22: c-s: c1,c2,c3
                           [False, False, True], # sample 22: t-a: c1,c2,c3
                           [False, False, True]], # sample 22: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 23: cit: c1,c2,c3
                           [False, False, False], # sample 23: c-s: c1,c2,c3
                           [False, False, True], # sample 23: t-a: c1,c2,c3
                           [True, True, True]], # sample 23: i-e: c1,c2,c3

                          [[False, False, False], # sample 24: cit: c1,c2,c3
                           [False, False, False], # sample 24: c-s: c1,c2,c3
                           [False, False, False], # sample 24: t-a: c1,c2,c3
                           [False, False, False]], # sample 24: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 25: cit: c1,c2,c3
                           [False, False, False], # sample 25: c-s: c1,c2,c3
                           [False, False, True], # sample 25: t-a: c1,c2,c3
                           [False, False, True]], # sample 25: i-e: c1,c2,c3

                          [[False, False, False], # sample 26: cit: c1,c2,c3
                           [False, False, False], # sample 26: c-s: c1,c2,c3
                           [False, False, False], # sample 26: t-a: c1,c2,c3
                           [True, False, False]], # sample 26: i-e: c1,c2,c3                

                          [[False, False, False], # sample 27: cit: c1,c2,c3
                           [False, False, False], # sample 27: c-s: c1,c2,c3
                           [False, False, True], # sample 27: t-a: c1,c2,c3
                           [False, False, True]], # sample 27: i-e: c1,c2,c3
                                
                          [[False, False, False], # sample 28: cit: c1,c2,c3,
                           [False, False, False], # sample 28: c-s: c1,c2,c3
                           [False, False, False], # sample 28: t-a: c1,c2,c3
                           [False, False, False]], # sample 28: i-e: c1,c2,c3

                          [[False, False, False], # sample 29: cit: c1,c2,c3
                           [False, False, False], # sample 29: c-s: c1,c2,c3
                           [False, False, False], # sample 29: t-a: c1,c2,c3
                           [False, False, True]], # sample 29: i-e: c1,c2,c3
                               
                          [[False, False, False], # sample 30: cit: c1,c2,c3
                           [False, False, True], # sample 30: c-s: c1,c2,c3
                           [False, False, True], # sample 30: t-a: c1,c2,c3
                           [False, False, False]], # sample 30: i-e: c1,c2,c3
                           ])


########################################################################################################################


contexts = ['In-Line Citations', 'Cond Sum.', 'Title-Abs.', 'Intro-Entity']
jurors = [[j1, j1_confidence], [j2, j2_confidence], [j3, j3_confidence], [j4, j4_confidence]]

n = j1.shape[0]

def compute_weights(conf):
    n_high_conf = len(conf) - conf.sum()
    n_splits = len(conf) + n_high_conf
    weights = []
    for i in conf:
        if i == True:
            weights.append(1/n_splits)
        else:
            weights.append(2/n_splits)
    return weights

def get_data_context(i, ind = 0):
    output = []
    for s in range(n):
        l = [jurors[j][ind][s, i, :] for j in range(len(jurors))]
        output.append(l)
    return np.array(output)

# compute scores
for i, c in enumerate(contexts):
    data = get_data_context(i, ind = 0)
    confidence_levels = get_data_context(i, ind = 1)
    ratings = []
    for s in range(data.shape[0]):
        sample_data = data[s, :, :]
        sample_confidence = confidence_levels[s, :, :]
        scores = []
        for crit in range(sample_data.shape[1]):
            data_crit = sample_data[:, crit]
            conf_crit = sample_confidence[:, crit]
            weights = compute_weights(conf_crit) # if box checked -> get only half of the weight
            score = np.dot(data_crit, weights)
            scores.append(score)
        ratings.append(scores)
    final_score = np.mean(ratings, axis = 0) # average across the qualitative metrics to give an aggregate score
    std_error = sem(ratings, axis = 0) # standard error of mean estimates across samples
    overall_score = np.mean(final_score)
    print(f'Human Scores for {c}: \n{final_score} with respective standard error estimates {std_error}\n{overall_score:.3f}\n')


# print correlation coefficients of jurors
juror_ratings = np.array([jurors[i][0].flatten() for i in range(len(jurors))])
print(juror_ratings, juror_ratings.shape)
print(np.corrcoef(np.array(juror_ratings)))
