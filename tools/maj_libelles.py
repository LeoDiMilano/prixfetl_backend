import psycopg2
import pandas as pd
import re


class P_M_S_Updater:
    def __init__(self, db_config):
        """
        Initialise la connexion PostgreSQL.
        :param db_config: Dictionnaire contenant les paramètres de connexion à la base PostgreSQL.
        """
        self.db_config = db_config

    def get_postgres_connection(self):
        """
        Crée et retourne une connexion PostgreSQL.
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except psycopg2.Error as e:
            print(f"Erreur de connexion à la base PostgreSQL : {e}")
            raise

    @staticmethod
    def extraire_calibre(libelle):
        for sep in ['-', '/', '+']:
            pos_sep = libelle.rfind(sep)
            while pos_sep != -1:
                if (pos_sep > 0 and libelle[pos_sep - 1].isdigit()) or \
                   (pos_sep < len(libelle) - 1 and libelle[pos_sep + 1].isdigit()):
                    pos_before = libelle.rfind(' ', 0, pos_sep)
                    pos_after = libelle.find(' ', pos_sep)
                    if pos_before != -1 and pos_after != -1:
                        return libelle[pos_before + 1:pos_after].strip()
                    elif pos_before != -1:
                        return libelle[pos_before + 1:].strip()
                    elif pos_after != -1:
                        return libelle[:pos_after].strip()
                pos_sep = libelle.rfind(sep, 0, pos_sep)
        return ''

    @staticmethod
    def extraire_espece(libelle):
        # si le libelle commence par jab, on retourne POMME
        if re.match(r'jab', libelle):
            return 'POMME'
        else:
            return libelle.split(' ')[0] if libelle else ''

    @staticmethod
    def extraire_variete(libelle):
        # si le libelle commence par jab, on on cherche le texte entre le premier - et _ en majuscule
        if re.match(r'jab', libelle):
            pos_sep = libelle.find('-')
            pos_underscore = libelle.find('_')
            if pos_sep != -1 and pos_underscore != -1:
                return libelle[pos_sep + 1:pos_underscore].upper()
            elif pos_sep != -1:
                return libelle[pos_sep + 1:].upper()
            elif pos_underscore != -1:
                return libelle[:pos_underscore].upper()
            else:
                return ''
        else:
            mots = libelle.split(' ')
            return mots[1].upper() if len(mots) > 1 else ''

    @staticmethod
    def extraire_emballage(libelle):
        if not libelle:
            return ''
        # Normaliser le libellé
        libelle = libelle.lower().strip()

        if 'caisse' in libelle:
            return 'caisse'
        if 'plateau 1rg' in libelle:
            return 'plateau 1rg'
        if 'plateau 2rg' in libelle:
            return 'plateau 2rg'
        if 'sachet' in libelle:
            return 'sachet'
        if 'bushel' in libelle:
            return 'bushel'
        if 'benne' in libelle:
            return 'benne'
        if 'barq' in libelle:
            return 'barquette'
        if 'filet' in libelle:
            return 'filet'
        if 'carton' in libelle:
            return 'carton'
        if 'plateau' in libelle:
            return 'plateau'
        if 'colis' in libelle:
            return 'colis'
        if 'vrac' in libelle:
            return 'vrac'
        if 'cagette' in libelle:
            return 'cagette'
        if 'conteneur' in libelle:
            return 'conteneur'

        return ''

    @staticmethod
    def extraire_origine(libelle):
        # si le libelle commence par jab, prendre le texte après _ en majuscule
        if re.match(r'jab', libelle):
            pos_underscore = libelle.find('_')
            if pos_underscore != -1:
                # si c'est kraj, on retourne PL
                if 'kraj' in libelle:
                    return 'PL'
                else:
                    return libelle[pos_underscore + 1:].upper()
            else:
                return ''
        else:
            if any(kw in libelle for kw in ['France', 'DOM','Sud-Est', 'Corse', 'Val de Loire', 'Roussillon','Martinique', 'Guadeloupe',
                'Rhône-Alpes', 'Sud-Ouest', 'Alsace', 'Savoie','Centre-Ouest','Limousin','Deux-Sèvres','Nord-Picardie']):
                return 'FRANCE'
            if 'kraj' in libelle:
                return 'PL'            
            if 'Nouvelle-Zélande' in libelle:
                return 'NZ'
            if 'Grèce' in libelle:
                return 'GRECE'
            if 'Afrique' in libelle:
                return 'AFRIQUE'
            if 'Espagne' in libelle:
                return 'ESPAGNE'
            if 'Maroc' in libelle:
                return 'MAROC'
            if 'Portugal' in libelle:
                return 'PORTUGAL'
            if 'Amérique' in libelle:
                return 'AMERIQUE'
            if 'Allemagne' in libelle:
                return 'ALLEMAGNE'
            if 'Italie' in libelle:
                return 'ITALIE'
            if 'U.E.' in libelle:
                return 'UE'
            if 'Tunisie' in libelle:
                return 'TUNISIE'
            if 'Chili' in libelle:
                return 'CHILI'
            if 'Israël' in libelle:
                return 'ISRAEL'
            if 'Belgique' in libelle:
                return 'BELGIQUE'
            if 'Egypte' in libelle:
                return 'EGYPTE'
            if 'Colombie' in libelle:
                return 'COLOMBIE'
            if 'Pays-Bas' in libelle:
                return 'PAYS-BAS'
            if 'hors Fr.' in libelle:
                return 'HORS FR'
            return ''

    @staticmethod
    def extraire_categorie(libelle):
        if 'biologique' in libelle:
            return 'bio'
        if 'industrie' in libelle:
            return 'industrie'
        if 'cat.II' in libelle:
            return 'cat.II'
        if 'cat.I' in libelle:
            return 'cat.I'
        return ''


    def ajouter_et_mettre_a_jour(self):
        try:
            conn = self.get_postgres_connection()
            cursor = conn.cursor()

            # Ajouter les valeurs manquantes dans PRODUIT_MARCHE_STADE
            cursor.execute("""
                INSERT INTO PRODUIT_MARCHE_STADE (MARCHE, STADE, LIBELLE_PRODUIT)
                SELECT DISTINCT MARCHE, STADE, LIBELLE_PRODUIT
                FROM COTATIONS_RNM_JOURNALIERES
                ON CONFLICT DO NOTHING
            """)

            # Récupérer les lignes à mettre à jour
            cursor.execute("""
                SELECT MARCHE, STADE, LIBELLE_PRODUIT, ESPECE, VARIETE, CALIBRE, EMBALLAGE, ORIGINE, CATEGORIE
                FROM PRODUIT_MARCHE_STADE
            """)
            lignes = cursor.fetchall()

            for row in lignes:
                marche, stade, libelle, espece, variete, calibre, emballage, origine, categorie = row

                if libelle:
                    nouveau_espece = self.extraire_espece(libelle) if not espece else espece
                    nouvelle_variete = self.extraire_variete(libelle) if not variete else variete
                    nouveau_calibre = self.extraire_calibre(libelle) if not calibre else calibre
                    nouvel_emballage = self.extraire_emballage(libelle) if not emballage else emballage
                    nouvelle_origine = self.extraire_origine(libelle) if not origine else origine
                    nouvelle_categorie = self.extraire_categorie(libelle) if not categorie else categorie

                    cursor.execute("""
                        UPDATE PRODUIT_MARCHE_STADE
                        SET ESPECE = %s, VARIETE = %s, CALIBRE = %s, EMBALLAGE = %s, ORIGINE = %s, CATEGORIE = %s
                        WHERE MARCHE = %s AND STADE = %s AND LIBELLE_PRODUIT = %s
                    """, (nouveau_espece, nouvelle_variete, nouveau_calibre, nouvel_emballage, nouvelle_origine, nouvelle_categorie, marche, stade, libelle))

            conn.commit()
        except psycopg2.Error as e:
            print(f"Erreur d'accès à la base PostgreSQL : {e}")
        finally:
            conn.close()
            print("Ajout et mise à jour terminés.")

    def exporter_vers_excel(self, query, output_path):
        try:
            conn = self.get_postgres_connection()
            df = pd.read_sql_query(query, conn)
            df.to_excel(output_path, index=False)
            print(f"Exportation réussie vers : {output_path}")
        except Exception as e:
            print(f"Erreur lors de l'exportation : {e}")
        finally:
            conn.close()


if __name__ == "__main__":
    db_config = {
        "host": "prixfetl_postgres",
        "port": 5432,
        "database": "IAFetL",
        "user": "prixfetl",
        "password": "Leumces123"
    }

    updater = P_M_S_Updater(db_config)

    # Exécuter les mises à jour
    updater.ajouter_et_mettre_a_jour()

    # Exporter les données vers Excel
    query = "SELECT * FROM PRODUIT_MARCHE_STADE"
    output_path = "/app/tests/resultats_produit_marche_stade.xlsx"
    updater.exporter_vers_excel(query, output_path)
