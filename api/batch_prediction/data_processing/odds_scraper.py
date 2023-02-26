from selenium import webdriver
from selenium.webdriver.common.by import By

class OddsScraper(object):
    def __init__(self, required_games):
        self.url = 'https://oddspedia.com/basketball/usa/nba#odds'
        self.required_games = required_games
        self.scraped_games = []
        self.game_keys = set()
        
    def _open(self):
        self.driver = webdriver.Chrome()
    
    def _close(self):
        self.driver.close()
        
    def _get_main_page(self):
        self.driver.get(self.url)
        self.driver.implicitly_wait(8)

        # reject notifications
        try:
            cancel_button = self.driver.find_element(By.ID, 'onesignal-slidedown-cancel-button')
            cancel_button.click()
        except:
            None

        # select all available bookmakers
        bookmaker_select = self.driver.find_element(By.CLASS_NAME, 'dropdown__toggle')
        bookmaker_select.click()
        all_bookmakers = self.driver.find_element(By.CLASS_NAME, 'bm-dropdown__select-all')
        all_bookmakers.click()
        
    def _get_content(self):
        odds_div = self.driver.find_element(By.CLASS_NAME, 'ml__wrap')
        games = odds_div.find_elements(By.CLASS_NAME, 'match-list-item')

        game_dates = odds_div.find_elements(By.CLASS_NAME, 'match-list-headline-league')
        dates = [date.text.split(' - ')[0] for date in game_dates]
        date_ordering = [date.rect['y'] for date in game_dates]

        i = 0
        for game in games:
            game_info = {'home_team': None, 'away_team':None, 
                     'home_odds':None, 'away_odds':None, 
                     'home_score':None, 'away_score':None,
                     'date':None}

            # team names
            teams = game.find_element(By.TAG_NAME, 'a').get_attribute('title').split(' - ')
            game_info['home_team'], game_info['away_team'] = teams[0], teams[1]
            
            game_key = f'{teams[0]} vs. {teams[1]}'
            if game_key not in self.required_games:
                continue
            if game_key in self.game_keys:
                continue

            # odds
            odds = game.find_elements(By.CLASS_NAME, 'odd__value')
            if len(odds) > 0:
                game_info['home_odds'], game_info['away_odds'] = odds[0].text, odds[1].text

            # score
            score = game.find_elements(By.CLASS_NAME, 'match-score-result__score')
            if len(score) > 0:
                game_info['home_score'], game_info['away_score'] = score[-2].text, score[-1].text

            # date
            if i+1 < len(date_ordering):
                if game.rect['y'] > date_ordering[i+1]:
                    i += 1
            game_info['date'] = dates[i]

            self.scraped_games.append(game_info)
            self.game_keys.add(game_key)
            
    def _change_page(self):
        left_button = self.driver.find_element(By.CLASS_NAME, 'ml-pagination__btn')
        left_button.click()
            
    def scrape(self):
        self._open()
        self._get_main_page()
        
        self._get_content()
        self._change_page()
        self._get_content()
        
        self._close()
        return self.scraped_games
    
    def get_odds(self):
        if self.scraped_games == []:
            self.scrape()
        odds = {}
        for game_info in self.scraped_games:
            game_key = game_info['home_team'] + ' vs. ' + game_info['away_team']
            if game_key not in self.required_games:
                continue
            home_odds, away_odds = game_info['home_odds'], game_info['away_odds']
            odds[game_key] = {'home_odds': home_odds, 'away_odds': away_odds}
        return odds

