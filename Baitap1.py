import pandas as pd
from bs4 import BeautifulSoup
import requests

def fetch_player_stats(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        player_table = soup.find('table', {'id': 'stats_standard'})
        if not player_table:
            print("Standard stats table not found.")
            return pd.DataFrame()

        tbody = player_table.find('tbody')
        if not tbody:
            print("Standard stats table body not found.")
            return pd.DataFrame()

        players_data = []
        for row in tbody.find_all('tr'):
            if row.find('th', {'scope': 'row'}):
                player_stats = {}
                player_stats['Nationality'] = row.find('td', {'data-stat': 'nationality'}).text if row.find('td', {'data-stat': 'nationality'}) else 'N/a'
                player_stats['Name'] = row.find('th', {'data-stat': 'player'}).text if row.find('th', {'data-stat': 'player'}) else 'N/a'
                player_stats['Squad'] = row.find('td', {'data-stat': 'team'}).text if row.find('td', {'data-stat': 'team'}) else 'N/a'
                player_stats['Position'] = row.find('td', {'data-stat': 'pos'}).text if row.find('td', {'data-stat': 'pos'}) else 'N/a'
                player_stats['Age'] = row.find('td', {'data-stat': 'age'}).text if row.find('td', {'data-stat': 'age'}) else 'N/a'
                player_stats['MP'] = row.find('td', {'data-stat': 'games'}).text if row.find('td', {'data-stat': 'games'}) else 'N/a'
                player_stats['Starts'] = row.find('td', {'data-stat': 'games_starts'}).text if row.find('td', {'data-stat': 'games_starts'}) else 'N/a'
                player_stats['Min'] = row.find('td', {'data-stat': 'minutes'}).text if row.find('td', {'data-stat': 'minutes'}) else 'N/a'

                minutes_played = int(player_stats['Min']) if player_stats['Min'].isdigit() else 0
                if minutes_played > 90:
                    player_stats['Gls'] = row.find('td', {'data-stat': 'goals'}).text if row.find('td', {'data-stat': 'goals'}) else 'N/a'
                    player_stats['Ast'] = row.find('td', {'data-stat': 'assists'}).text if row.find('td', {'data-stat': 'assists'}) else 'N/a'
                    player_stats['Yel'] = row.find('td', {'data-stat': 'cards_yellow'}).text if row.find('td', {'data-stat': 'cards_yellow'}) else 'N/a'
                    player_stats['Red'] = row.find('td', {'data-stat': 'cards_red'}).text if row.find('td', {'data-stat': 'cards_red'}) else 'N/a'
                    player_stats['xG'] = row.find('td', {'data-stat': 'xg'}).text if row.find('td', {'data-stat': 'xg'}) else 'N/a'
                    player_stats['xAG'] = row.find('td', {'data-stat': 'xa'}).text if row.find('td', {'data-stat': 'xa'}) else 'N/a'
                    player_stats['PrgC'] = row.find('td', {'data-stat': 'passes_completed_progressive_distance'}).text if row.find('td', {'data-stat': 'passes_completed_progressive_distance'}) else 'N/a'
                    player_stats['PrgP'] = row.find('td', {'data-stat': 'passes_progressive_distance'}).text if row.find('td', {'data-stat': 'passes_progressive_distance'}) else 'N/a'
                    player_stats['PrgR'] = row.find('td', {'data-stat': 'dribbles_completed_progressive_distance'}).text if row.find('td', {'data-stat': 'dribbles_completed_progressive_distance'}) else 'N/a'
                    player_stats['Gls90'] = row.find('td', {'data-stat': 'goals_per90'}).text if row.find('td', {'data-stat': 'goals_per90'}) else 'N/a'
                    player_stats['Ast90'] = row.find('td', {'data-stat': 'assists_per90'}).text if row.find('td', {'data-stat': 'assists_per90'}) else 'N/a'
                    player_stats['xG90'] = row.find('td', {'data-stat': 'xg_per90'}).text if row.find('td', {'data-stat': 'xg_per90'}) else 'N/a'
                    player_stats['xAG90'] = row.find('td', {'data-stat': 'xa_per90'}).text if row.find('td', {'data-stat': 'xa_per90'}) else 'N/a'

                    player_link = row.find('a', string=player_stats['Name'])
                    if player_link and 'href' in player_link.attrs:
                        player_url = f"https://fbref.com{player_link['href']}"
                        player_soup = BeautifulSoup(requests.get(player_url, headers=headers).content, 'html.parser')

                        # Goalkeeper Stats
                        if player_stats['Position'] == 'GK':
                            keeper_table = player_soup.find('table', {'id': 'stats_keeper'})
                            if keeper_table and keeper_table.find('tbody') and keeper_table.find('tbody').find('tr'):
                                gk_row = keeper_table.find('tbody').find('tr')
                                player_stats['GA90'] = gk_row.find('td', {'data-stat': 'goals_against_gk_per90'}).text if gk_row.find('td', {'data-stat': 'goals_against_gk_per90'}) else 'N/a'
                                player_stats['Save%'] = gk_row.find('td', {'data-stat': 'save_pct'}).text if gk_row.find('td', {'data-stat': 'save_pct'}) else 'N/a'
                                player_stats['CS%'] = gk_row.find('td', {'data-stat': 'clean_sheets_pct'}).text if gk_row.find('td', {'data-stat': 'clean_sheets_pct'}) else 'N/a'
                                player_stats['Pen Save%'] = gk_row.find('td', {'data-stat': 'penalty_save_pct'}).text if gk_row.find('td', {'data-stat': 'penalty_save_pct'}) else 'N/a'
                            else:
                                player_stats['GA90'] = 'N/a'
                                player_stats['Save%'] = 'N/a'
                                player_stats['CS%'] = 'N/a'
                                player_stats['Pen Save%'] = 'N/a'
                        else:
                            player_stats['GA90'] = 'N/a'
                            player_stats['Save%'] = 'N/a'
                            player_stats['CS%'] = 'N/a'
                            player_stats['Pen Save%'] = 'N/a'

                        # Shooting Stats
                        shooting_table = player_soup.find('table', {'id': 'stats_shooting'})
                        if shooting_table and shooting_table.find('tbody') and shooting_table.find('tbody').find('tr'):
                            shoot_row = shooting_table.find('tbody').find('tr')
                            player_stats['SoT%'] = shoot_row.find('td', {'data-stat': 'shots_on_target_pct'}).text if shoot_row.find('td', {'data-stat': 'shots_on_target_pct'}) else 'N/a'
                            player_stats['SoT90'] = shoot_row.find('td', {'data-stat': 'shots_on_target_per90'}).text if shoot_row.find('td', {'data-stat': 'shots_on_target_per90'}) else 'N/a'
                            player_stats['G/Sh'] = shoot_row.find('td', {'data-stat': 'goals_per_shot'}).text if shoot_row.find('td', {'data-stat': 'goals_per_shot'}) else 'N/a'
                            player_stats['Dist'] = shoot_row.find('td', {'data-stat': 'average_shot_distance'}).text if shoot_row.find('td', {'data-stat': 'average_shot_distance'}) else 'N/a'
                        else:
                            player_stats['SoT%'] = 'N/a'
                            player_stats['SoT90'] = 'N/a'
                            player_stats['G/Sh'] = 'N/a'
                            player_stats['Dist'] = 'N/a'

                        # Passing Stats
                        passing_table = player_soup.find('table', {'id': 'stats_passing'})
                        if passing_table and passing_table.find('tbody') and passing_table.find('tbody').find('tr'):
                            pass_row = passing_table.find('tbody').find('tr')
                            player_stats['Cmp'] = pass_row.find('td', {'data-stat': 'passes_completed'}).text if pass_row.find('td', {'data-stat': 'passes_completed'}) else 'N/a'
                            player_stats['Cmp%'] = pass_row.find('td', {'data-stat': 'passes_pct'}).text if pass_row.find('td', {'data-stat': 'passes_pct'}) else 'N/a'
                            player_stats['TotDist'] = pass_row.find('td', {'data-stat': 'passes_total_distance'}).text if pass_row.find('td', {'data-stat': 'passes_total_distance'}) else 'N/a'
                            player_stats['Cmp%_Short'] = pass_row.find('td', {'data-stat': 'passes_short_pct'}).text if pass_row.find('td', {'data-stat': 'passes_short_pct'}) else 'N/a'
                            player_stats['Cmp%_Med'] = pass_row.find('td', {'data-stat': 'passes_medium_pct'}).text if pass_row.find('td', {'data-stat': 'passes_medium_pct'}) else 'N/a'
                            player_stats['Cmp%_Long'] = pass_row.find('td', {'data-stat': 'passes_long_pct'}).text if pass_row.find('td', {'data-stat': 'passes_long_pct'}) else 'N/a'
                            player_stats['KP'] = pass_row.find('td', {'data-stat': 'assisted_shots'}).text if pass_row.find('td', {'data-stat': 'assisted_shots'}) else 'N/a'
                            player_stats['1/3'] = pass_row.find('td', {'data-stat': 'passes_into_final_third'}).text if pass_row.find('td', {'data-stat': 'passes_into_final_third'}) else 'N/a'
                            player_stats['PPA'] = pass_row.find('td', {'data-stat': 'passes_into_penalty_area'}).text if pass_row.find('td', {'data-stat': 'passes_into_penalty_area'}) else 'N/a'
                            player_stats['CrsPA'] = pass_row.find('td', {'data-stat': 'crosses_into_penalty_area'}).text if pass_row.find('td', {'data-stat': 'crosses_into_penalty_area'}) else 'N/a'
                            player_stats['PrgP_Pass'] = pass_row.find('td', {'data-stat': 'passes_progressive_distance'}).text if pass_row.find('td', {'data-stat': 'passes_progressive_distance'}) else 'N/a'
                        else:
                            player_stats['Cmp'] = 'N/a'
                            player_stats['Cmp%'] = 'N/a'
                            player_stats['TotDist'] = 'N/a'
                            player_stats['Cmp%_Short'] = 'N/a'
                            player_stats['Cmp%_Med'] = 'N/a'
                            player_stats['Cmp%_Long'] = 'N/a'
                            player_stats['KP'] = 'N/a'
                            player_stats['1/3'] = 'N/a'
                            player_stats['PPA'] = 'N/a'
                            player_stats['CrsPA'] = 'N/a'
                            player_stats['PrgP_Pass'] = 'N/a'

                        # Goal and Shot Creation
                        gca_table = player_soup.find('table', {'id': 'stats_gca'})
                        if gca_table and gca_table.find('tbody') and gca_table.find('tbody').find('tr'):
                            gca_row = gca_table.find('tbody').find('tr')
                            player_stats['SCA'] = gca_row.find('td', {'data-stat': 'sca_total'}).text if gca_row.find('td', {'data-stat': 'sca_total'}) else 'N/a'
                            player_stats['SCA90'] = gca_row.find('td', {'data-stat': 'sca_per90'}).text if gca_row.find('td', {'data-stat': 'sca_per90'}) else 'N/a'
                            player_stats['GCA'] = gca_row.find('td', {'data-stat': 'gca_total'}).text if gca_row.find('td', {'data-stat': 'gca_total'}) else 'N/a'
                            player_stats['GCA90'] = gca_row.find('td', {'data-stat': 'gca_per90'}).text if gca_row.find('td', {'data-stat': 'gca_per90'}) else 'N/a'
                        else:
                            player_stats['SCA'] = 'N/a'
                            player_stats['SCA90'] = 'N/a'
                            player_stats['GCA'] = 'N/a'
                            player_stats['GCA90'] = 'N/a'

                        # Defensive Actions
                        defense_table = player_soup.find('table', {'id': 'stats_defense'})
                        if defense_table and defense_table.find('tbody') and defense_table.find('tbody').find('tr'):
                            def_row = defense_table.find('tbody').find('tr')
                            player_stats['Tkl'] = def_row.find('td', {'data-stat': 'tackles'}).text if def_row.find('td', {'data-stat': 'tackles'}) else 'N/a'
                            player_stats['TklW'] = def_row.find('td', {'data-stat': 'tackles_won'}).text if def_row.find('td', {'data-stat': 'tackles_won'}) else 'N/a'
                            player_stats['Att'] = def_row.find('td', {'data-stat': 'challenge_lost'}).text if def_row.find('td', {'data-stat': 'challenge_lost'}) else 'N/a'
                            player_stats['Lost'] = def_row.find('td', {'data-stat': 'challenge_lost'}).text if def_row.find('td', {'data-stat': 'challenge_lost'}) else 'N/a' # Corrected data-stat
                            player_stats['Blocks'] = def_row.find('td', {'data-stat': 'blocks'}).text if def_row.find('td', {'data-stat': 'blocks'}) else 'N/a'
                            player_stats['Sh'] = def_row.find('td', {'data-stat': 'blocked_shots'}).text if def_row.find('td', {'data-stat': 'blocked_shots'}) else 'N/a'
                            player_stats['Pass'] = def_row.find('td', {'data-stat': 'blocked_passes'}).text if def_row.find('td', {'data-stat': 'blocked_passes'}) else 'N/a'
                            player_stats['Int'] = def_row.find('td', {'data-stat': 'interceptions'}).text if def_row.find('td', {'data-stat': 'interceptions'}) else 'N/a'
                        else:
                            player_stats['Tkl'] = 'N/a'
                            player_stats['TklW'] = 'N/a'
                            player_stats['Att'] = 'N/a'
                            player_stats['Lost'] = 'N/a'
                            player_stats['Blocks'] = '